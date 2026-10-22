function data=torchAdaptiveData(s)
% Schema-3 metadata and leaf-ordered fields shared by all output writers.
c=torchAdaptiveValidate(s.config);d=c.dimension;n=size(s.mesh.keys,1);U=s.state;
assert(isequal(size(U),[d+2,n])&&torchAdaptivePhysical(U),'Invalid adaptive state');
assert(isscalar(s.time)&&isfinite(s.time)&&s.time>=0&&s.time<=c.t_end,'Invalid time');
for key={'steps','regrids','mesh_revision'}
    v=s.(key{1});assert(isscalar(v)&&isfinite(v)&&v>=0&&v==fix(v),'Invalid counters');
end
assert(s.steps<=c.max_steps&&s.mesh_revision<=s.regrids&&islogical(s.budget_limited)&&isscalar(s.budget_limited),'Invalid mesh counters');
assert(ischar(s.run_id)&&~isempty(s.run_id),'Invalid run identity');
for key={'initial_integral','boundary_integral','source_integral'}
    v=s.(key{1});assert(numel(v)==d+2&&all(isfinite(v(:))),'Invalid balances');
end
assert(s.initial_integral(1)>0,'Invalid initial mass');
if c.reactive,assert(isequal(size(s.lam),[1,n])&&all(isfinite(s.lam))&&all(s.lam>=0&s.lam<=1),'Invalid lambda');
else,assert(isempty(s.lam),'Inert state has composition');end
assert(isempty(s.beta),'MATLAB does not support ZND');
[P,~]=torchAdaptivePrimitive(U,s.lam,c);a=torchCoordinates(c);names=[{'rho'},cellfun(@(x)['momentum_' x],a,'UniformOutput',false),{'energy'}];f=struct;
for k=1:d+2,f.(names{k})=U(k,:)';end
f.pressure=P(end,:)';for k=1:d,f.(['velocity_' a{k}])=P(k+1,:)';end
f.refinement_level=double(s.mesh.keys(:,1));f.cell_volume=s.mesh.volumes(:);
if c.reactive,f.lambda=s.lam(:);end
if ~strcmp(c.geometry,'cartesian')
    v=zeros(3,n);v(1:d,:)=P(2:end-1,:);x=s.mesh.centers;
    if strcmp(c.geometry,'cylindrical')&&d==2,v(3,:)=v(2,:);v(2,:)=0;
    elseif d>=2
        t=x(:,2)';w=v;
        if strcmp(c.geometry,'spherical')
            phi=zeros(1,n);if d==3,phi=x(:,3)';end
            v(1,:)=(w(1,:).*sin(t)+w(2,:).*cos(t)).*cos(phi)-w(3,:).*sin(phi);
            v(2,:)=(w(1,:).*sin(t)+w(2,:).*cos(t)).*sin(phi)+w(3,:).*cos(phi);
            v(3,:)=w(1,:).*cos(t)-w(2,:).*sin(t);
        else,v(1,:)=w(1,:).*cos(t)-w(2,:).*sin(t);v(2,:)=w(1,:).*sin(t)+w(2,:).*cos(t);end
    end
    xyz='xyz';for k=1:3,f.(['velocity_cartesian_' xyz(k)])=v(k,:)';end
end
meta=struct('format','TORCH checkpoint','schema_version',3,'solver_version','2.0','geometry',c.geometry,'units','SI','layout','leaf-level-z-y-x','config',c,'run_id',s.run_id,'time',s.time,'steps',s.steps,'initial_integral',s.initial_integral,'boundary_integral',s.boundary_integral,'source_integral',s.source_integral,'current_integral',torchSessionIntegral(s),'cell_count',n,'regrids',s.regrids,'mesh_revision',s.mesh_revision,'budget_limited',s.budget_limited);
data=struct('metadata',meta,'names',{names},'fields',f);
end
