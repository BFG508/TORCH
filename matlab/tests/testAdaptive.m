function report=testAdaptive()
% Exact metric partitions, balance, periodic closure and analytic advection.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
for d=1:3
    for geometry={'cartesian','cylindrical','annular','spherical'}
        counts=[8,4,4];counts=counts(1:d);P=ones([d+2,fliplr(counts)]);flat=reshape(P,d+2,[]);flat(2:end-1,:)=0;P=reshape(flat,size(P));
        origins=zeros(1,d);if ~strcmp(geometry{1},'cartesian'),origins=[.4,.3,.2];origins=origins(1:d);end
        s=torchCreateSession(P,ones(1,d),'geometry',geometry{1},'origins',origins,'t_end',.02,'adaptive',struct('max_level',2));
        total=torchSessionIntegral(s);volume=sum(s.mesh.volumes);s=torchAdaptiveRegrid(s,s.mesh.centers(:,1)<.8,false);
        assert(abs(sum(s.mesh.volumes)-volume)<1e-12,'Metric partition changed');assert(max(abs(torchSessionIntegral(s)-total))<1e-12,'Refinement lost conservation');
        s=torchRunSession(s,'step_limit',2);velocity=s.state(2:end-1,:);assert(max(abs(velocity(:)))<1e-13,'Rest state moved');
        assert(max(abs(torchSessionIntegral(s)-s.initial_integral+s.boundary_integral-s.source_integral))<2e-12,'Adaptive balance mismatch');
        % Capacity is a hard limit, with no partial mesh mutation.
        if strcmp(geometry{1},'cartesian')
            capacity=prod(counts);a=torchCreateSession(P,ones(1,d),'adaptive',struct('max_cells',capacity));keys=a.mesh.keys;
            a=torchAdaptiveRegrid(a,true(capacity,1),false);assert(a.budget_limited&&isequal(a.mesh.keys,keys),'Capacity transaction failed');
        end
    end
end
for d=1:2
    for geometry={'cylindrical','spherical'}
        P=ones([d+2,repmat(6,1,d)]);flat=reshape(P,d+2,[]);flat(2:end-1,:)=0;P=reshape(flat,size(P));L=ones(1,d);
        if d==2&&strcmp(geometry{1},'spherical'),L(2)=pi;end
        s=torchCreateSession(P,L,'geometry',geometry{1},'t_end',.01,'adaptive',struct('max_level',1));s=torchAdaptiveRegrid(s,s.mesh.low(:,1)==0,false);s=torchRunSession(s);velocity=s.state(2:end-1,:);
        assert(max(abs(velocity(:)))<2e-14,'Adaptive rest failed at origin/pole');
    end
end
errors=zeros(1,3);sizes=[16,32,64];
for k=1:3
    n=sizes(k);x=((1:n)-.5)/n;P=[1+.2*sin(pi/n)/(pi/n)*sin(2*pi*x);.5*ones(1,n);ones(1,n)];
    s=torchCreateSession(P,1,'boundaries',{'Periodic'},'t_end',.1,'adaptive',struct('max_level',1,'refine_threshold',1,'coarsen_threshold',0,'regrid_every',10000));
    s=torchAdaptiveRegrid(s,s.mesh.centers(:,1)<.5,false);s=torchRunSession(s);x=s.mesh.centers(:,1)';w=s.mesh.widths(:,1)';exact=1+.2*sin(pi*w)./(pi*w).*sin(2*pi*(x-.5*s.time));errors(k)=sum(abs(s.state(1,:)-exact).*w);
end
orders=log2(errors(1:end-1)./errors(2:end));assert(min(orders)>1.8,'AMR entropy advection lost second order');
report=struct('entropy_sizes',sizes,'entropy_density_l1',errors,'entropy_orders',orders,'passed',true);fprintf('PASS MATLAB AMR metrics, capacity, conservation and analytic advection\n');
end
