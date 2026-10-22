function report = torchBenchmark(varargin)
% Warmed median timings and structural grid bytes; no machine-dependent gate.
% Run each source checkout in a fresh process for a fair before/after study.
p=inputParser;addParameter(p,'quick',false,@islogical);addParameter(p,'steps',5);
addParameter(p,'repeats',3);addParameter(p,'warmup',2);addParameter(p,'label','working tree');
addParameter(p,'output','out/benchmark-matlab.json');parse(p,varargin{:});o=p.Results;
for v=[o.steps,o.repeats,o.warmup],assert(isfinite(v)&&v>=1&&v==fix(v),'Invalid benchmark count');end
sizes=[512,192,40,40,24];if o.quick,sizes=[128,96,24,24,16];end
dimensions=[1,2,3,3,3];geometries={'cartesian','cartesian','cartesian','spherical','annular'};rows=struct([]);
for mode=1:5
    d=dimensions(mode);n=sizes(mode);geometry=geometries{mode};reactive=mode==5;counts=repmat(n,1,d);
    values=ones(d+2,n^d);values(2:end-1,:)=.1;pressure=1;if reactive,pressure=1e6;end
    wave=pressure*(1+.05*sin(2*pi*((1:n)-.5)/n));values(end,:)=reshape(repmat(wave,n^(d-1),1),1,[]);
    P=reshape(values,[d+2,counts(end:-1:1)]);L=ones(1,d);origins=zeros(1,d);
    if ~strcmp(geometry,'cartesian'),L=[.5,1,1];origins=[.5,.3,0];end
    init=torchCreateSession(P,L,'origins',origins,'geometry',geometry,'reactive',reactive,'t_end',10);
    samples=zeros(1,o.repeats);single=samples;
    for trial=1:o.repeats
        s=torchRunSession(init,'step_limit',o.warmup);before=s.steps;start=tic;s=torchRunSession(s,'step_limit',o.steps);samples(trial)=toc(start)/o.steps;
        assert(s.steps-before==o.steps,'End time reached; use fewer benchmark steps');
        s=torchRunSession(init,'step_limit',o.warmup);start=tic;
        for step=1:o.steps,s=torchRunSession(s,'step_limit',1);end
        single(trial)=toc(start)/o.steps;
        assert(s.steps-o.warmup==o.steps,'End time reached; use fewer benchmark steps');
    end
    G=torchMetricGrid(init.config);details=whos('G');grid_bytes=details.bytes;
    residual=torchSessionIntegral(s)-s.initial_integral+s.boundary_integral-s.source_integral;
    row=struct('dimension',d,'geometry',geometry,'counts',counts,'reactive',reactive,...
        'measured_steps',o.steps,'warmup_steps',o.warmup,'repeats',o.repeats,...
        'seconds_per_step',median(samples),'seconds_per_step_samples',samples,...
        'repeated_call_seconds_per_step',median(single),'cell_updates_per_second',n^d/median(samples),...
        'metric_grid_bytes',grid_bytes,'relative_balance_linf',max(abs(residual)./max(abs(s.initial_integral),1))); %#ok<AGROW>
    if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
    fprintf('%dD %s reactive=%d: %.2f ms/step, %.2f MiB metric grid\n',d,geometry,reactive,1000*median(samples),grid_bytes/2^20);
end
report=struct('schema_version',1,'suite','TORCH session benchmark','label',o.label,'runtime',version,...
    'computer',computer,'memory_method','whos metric grid bytes; not peak or total process RAM',...
    'timing_method','median per-step wall time after warmup','results',rows);
if ~isempty(o.output),torchAtomicText(o.output,[jsonencode(report) newline]);fprintf('Report: %s\n',o.output);end
end
