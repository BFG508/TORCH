function data = torchSessionData(s)
% Portable checkpoint metadata and x-fastest fields, shared by all IO writers.
if isfield(s,'mesh'),data=torchAdaptiveData(s);return;end
c=s.config;dim=c.dimension;nvars=dim+2;h=c.lengths_xyz./c.counts_xyz;
flat=reshape(s.state,nvars,[]); gamma=c.gamma;
internal=flat(end,:)-.5*sum(flat(2:end-1,:).^2,1)./flat(1,:);
assert(all(isfinite(flat(:))) && all(flat(1,:)>1e-12) && all(internal>1e-12),'Cannot save nonphysical state');
assert(isfinite(s.time) && s.time>=0 && s.time<=c.t_end && s.steps>=0 && s.steps==fix(s.steps) && s.steps<=c.max_steps,'Invalid counters');
for key={'initial_integral','boundary_integral','source_integral'}
    value=s.(key{1});assert(numel(value)==nvars && all(isfinite(value(:))),'Invalid balances');
end
if c.reactive,gamma=(4.5-s.lam)./(3.5-s.lam);gamma=gamma(:)';end
primitive=flat;primitive(2:end-1,:)=flat(2:end-1,:)./flat(1,:);
primitive(end,:)=(gamma-1).*(flat(end,:)-.5*sum(flat(2:end-1,:).^2,1)./flat(1,:));
axes=torchCoordinates(c);names=[{'rho'},cellfun(@(a)['momentum_' a],axes,'UniformOutput',false),{'energy'}];
fields=struct;
for k=1:nvars,fields.(names{k})=flatten(flat(k,:),c);end
fields.pressure=flatten(primitive(end,:),c);
for k=1:dim,fields.(['velocity_' axes{k}])=flatten(primitive(k+1,:),c);end
if isfield(c,'geometry') && ~strcmp(c.geometry,'cartesian')
    velocity=torchCartesianVelocity(c,torchGridOrder(reshape(primitive,size(s.state)),c,'flatten'));
    namesXYZ='xyz';for k=1:3,fields.(['velocity_cartesian_' namesXYZ(k)])=velocity(k,:)';end
end
if ~isempty(s.lam),fields.lambda=flatten(s.lam(:)',c);end
if ~isempty(s.beta),fields.beta=flatten(s.beta(:)',c);end
current=torchSessionIntegral(s);geometry='cartesian';if isfield(c,'geometry'),geometry=c.geometry;end
meta=struct('format','TORCH checkpoint','schema_version',2,'solver_version','1.4',...
    'geometry',geometry,'units','SI','layout','x-fastest','config',c,'run_id',s.run_id,...
    'time',s.time,'steps',s.steps,'initial_integral',s.initial_integral,...
    'boundary_integral',s.boundary_integral,'source_integral',s.source_integral,'current_integral',current);
data=struct('metadata',meta,'names',{names},'fields',fields);
end

function vector=flatten(row,c)
if c.dimension==1,vector=row(:);
else
    A=reshape(row,c.counts_xyz(end:-1:1));
    A=permute(A,c.dimension:-1:1);vector=A(:);
end
end
