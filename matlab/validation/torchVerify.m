function report = torchVerify(varargin)
% Independent analytic mesh studies. Return report and atomically save JSON.
% torchVerify('quick',true,'output','out/verification-matlab.json')
p=inputParser;addParameter(p,'quick',false,@islogical);addParameter(p,'output','out/verification-matlab.json');parse(p,varargin{:});o=p.Results;
rows=struct([]);checks=struct([]);
for kernel={'hllc','metric'}
    for kind={'vortex','acoustic','shock'}
        name=kind{1};dimensions=1:3;meshes=[12,24,48];minimum_order=1;
        if strcmp(name,'vortex'),dimensions=2:3;meshes=[24,48,96];if o.quick,meshes=[16,32];end
        elseif strcmp(name,'shock'),meshes=[80,160,320];minimum_order=.6;if o.quick,meshes=[40,80];end
        elseif o.quick,meshes=[8,16];end
        for d=dimensions
            required_order=minimum_order;if ~o.quick&&strcmp(name,'acoustic')&&d==1,required_order=1.6;end
            errors=zeros(1,numel(meshes));ok=true;
            for j=1:numel(meshes)
                n=meshes(j);s=torchVerificationCase(name,n,d,kernel{1});initial=s.state;s=torchRunSession(s);
                if strcmp(name,'shock'),exact=initial;else,exact=torchReference(name,n,s.time,d);end
                difference=abs(s.state-exact);flat=reshape(difference,d+2,[]);current=torchSessionIntegral(s);
                residual=current-s.initial_integral+s.boundary_integral-s.source_integral;
                row=struct('case',name,'dimension',d,'kernel',kernel{1},'counts',s.config.counts_xyz,'time',s.time,'steps',s.steps,...
                    'density_l1',mean(flat(1,:)),'conservative_l1',mean(flat(:)),'balance_linf',max(abs(residual)./max(abs(s.initial_integral),1)),...
                    'plateau_linf',[],'shock_position_error',[]);
                if strcmp(name,'shock')
                    density=reshape(s.state,d+2,[]);rho=mean(reshape(density(1,:),[],n),1);x=((1:n)-.5)/n;
                    values=reshape(flat(1,:),[],n);mask=x<.3|x>.7;row.plateau_linf=max(max(values(:,mask)));
                    [~,position]=max(abs(diff(rho)));row.shock_position_error=abs(position/n-.5);
                    ok=ok&&row.plateau_linf<.03&&row.shock_position_error<2/meshes(1);
                end
                ok=ok&&row.balance_linf<2e-10;errors(j)=row.density_l1;
                if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
            end
            orders=nan(1,numel(meshes)-1);
            for j=1:numel(orders)
                if min(errors(j:j+1))>5e-13,orders(j)=log2(errors(j)/errors(j+1));end
                ok=ok&&(errors(j+1)<=5e-13||orders(j)>required_order);
            end
            check=struct('case',name,'dimension',d,'kernel',kernel{1},'observed_density_orders',orders,'passed',logical(ok));
            if isempty(checks),checks=check;else,checks(end+1)=check;end %#ok<AGROW>
            status='PASS';if ~ok,status='FAIL';end
            fprintf('%s %s %dD %s: density orders ',status,kernel{1},d,name);
            for order=orders,if isnan(order),fprintf('roundoff ');else,fprintf('%.2f ',order);end;end;fprintf('\n');
        end
    end
end
report=struct('schema_version',1,'suite','TORCH analytic verification','quick',o.quick,'runtime',version,'passed',all([checks.passed]),'checks',checks,'results',rows);
if ~isempty(o.output),torchAtomicText(o.output,[jsonencode(report) newline]);fprintf('Report: %s\n',o.output);end
end
