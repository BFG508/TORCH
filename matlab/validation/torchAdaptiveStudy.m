function report=torchAdaptiveStudy(varargin)
% Nonuniform analytic advection plus bounded CPU execution measurements.
p=inputParser;addParameter(p,'quick',false);addParameter(p,'workers',[1,2,4]);addParameter(p,'output','out/adaptive-matlab.json');parse(p,varargin{:});o=p.Results;
sizes=[16,32,64];if o.quick,sizes=[12,24];end;checks=struct([]);
for d=1:3
    errors=zeros(size(sizes));balances=errors;leaves=errors;
    for k=1:numel(sizes)
        n=sizes(k);counts=[n,repmat(4,1,d-1)];P=ones([d+2,fliplr(counts)]);flat=reshape(P,d+2,[]);flat(2:end-1,:)=0;flat(2,:)=.5;
        x=((1:n)-.5)/n;rho=1+.2*sin(pi/n)/(pi/n)*sin(2*pi*x);flat(1,:)=reshape(repmat(rho,prod(counts(2:end)),1),1,[]);P=reshape(flat,size(P));
        s=torchCreateSession(P,ones(1,d),'boundaries',repmat({'Periodic'},1,d),'t_end',.1,'adaptive',struct('max_level',1,'refine_threshold',1,'coarsen_threshold',0,'regrid_every',10000));
        s=torchAdaptiveRegrid(s,s.mesh.centers(:,1)<.5,false);s=torchRunSession(s);x=s.mesh.centers(:,1)';w=s.mesh.widths(:,1)';exact=1+.2*sin(pi*w)./(pi*w).*sin(2*pi*(x-.5*s.time));errors(k)=sum(abs(s.state(1,:)-exact).*s.mesh.volumes);leaves(k)=size(s.state,2);
        balances(k)=max(abs(torchSessionIntegral(s)-s.initial_integral+s.boundary_integral-s.source_integral));
    end
    orders=log2(errors(1:end-1)./errors(2:end));check=struct('dimension',d,'sizes',sizes,'leaves',leaves,'density_l1',errors,'orders',orders,'balance_linf',balances,'passed',min(orders)>1.7&&max(balances)<2e-12);checks=[checks,check]; %#ok<AGROW>
end
cases=[2,192,0;3,40,0;2,32,1;3,12,1];steps=5;repeats=3;if o.quick,cases(:,2)=[96;24;16;8];steps=2;repeats=1;end
rows=struct([]);torchWorkers(max(o.workers));
for i=1:size(cases,1)
    d=cases(i,1);n=cases(i,2);adaptive=logical(cases(i,3));P=ones([d+2,repmat(n,1,d)]);flat=reshape(P,d+2,[]);flat(2:end-1,:)=.1;x=((1:n)-.5)/n;pressure=1+.3*exp(-((x-.5)/.1).^2);flat(end,:)=reshape(repmat(pressure,n^(d-1),1),1,[]);P=reshape(flat,size(P));
    options=[];if adaptive,options=struct('max_level',1,'refine_threshold',.02,'coarsen_threshold',.005,'regrid_every',3,'max_cells',200000);end
    template=torchCreateSession(P,ones(1,d),'t_end',10,'face_boundaries',struct,'adaptive',options);
    for workers=o.workers
        torchWorkers(workers);samples=zeros(1,repeats);
        for k=1:repeats
            s=template;s.workers=workers;s=torchRunSession(s,'step_limit',2);start=tic;s=torchRunSession(s,'step_limit',steps);samples(k)=toc(start)/steps;
        end
        memory=whos('s');count=prod(s.config.counts_xyz);regrid=0;finest=count;
        if adaptive,count=size(s.mesh.keys,1);finest=(2*n)^d;start=tic;s=torchAdaptiveRegrid(s);regrid=toc(start);end
        balance=max(abs(torchSessionIntegral(s)-s.initial_integral+s.boundary_integral-s.source_integral));
        row=struct('dimension',d,'roots',repmat(n,1,d),'adaptive',adaptive,'workers',workers,'warmup_steps',2,'measured_steps',steps,'repeats',repeats,'seconds_per_step',median(samples),'samples',samples,'leaves',count,'finest_uniform_cells',finest,'leaf_fraction_of_finest',count/finest,'regrid_seconds',regrid,'session_bytes',memory.bytes,'balance_linf',balance);rows=[rows,row]; %#ok<AGROW>
    end
end
report=struct('schema_version',1,'suite','TORCH 2.0 AMR and CPU execution','runtime',version,'quick',o.quick,'passed',all([checks.passed]),'memory_scope','MATLAB whos session structure after warmed steps; excludes transient work arrays, PCT pools and process RSS. Not comparable to Python traced peaks.','timing_scope','Wall time including CFL, reconstruction, fluxes, scatter and scheduled regrids; excludes initial pool and topology construction.','checks',checks,'benchmarks',rows);
if ~isempty(o.output),torchAtomicText(o.output,[jsonencode(report) newline]);end
assert(report.passed,'Adaptive analytic verification failed');fprintf('PASS MATLAB AMR study; report: %s\n',o.output);
end
