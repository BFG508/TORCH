function s=torchReadAdaptive(path,m)
% Read a non-executable schema-3 dyadic leaf checkpoint, Python or MATLAB.
assert(m.schema_version==3&&strcmp(m.format,'TORCH checkpoint')&&strcmp(m.solver_version,'2.0')&&strcmp(m.layout,'leaf-level-z-y-x')&&strcmp(m.units,'SI'),'Unsupported adaptive schema');
c=torchAdaptiveValidate(m.config);assert(strcmp(m.geometry,c.geometry),'Geometry mismatch');d=c.dimension;n=m.cell_count;
assert(isscalar(n)&&isfinite(n)&&n==fix(n)&&n>0&&n<=c.amr.max_cells,'Invalid leaf count');
levels=dataset(path,'/mesh/levels',n,'int64');indices=dataset(path,'/mesh/indices',n*d,'int64');K=double([levels,reshape(indices,d,n)']);
M=torchAdaptiveMesh(c,K);assert(isequal(K,M.keys)&&all(K(:,1)>=c.amr.min_level),'Noncanonical or undersized leaves');
a=torchCoordinates(c);names=[{'rho'},cellfun(@(x)['momentum_' x],a,'UniformOutput',false),{'energy'}];U=zeros(d+2,n);
for k=1:d+2,U(k,:)=dataset(path,['/state/' names{k}],n,'double')';end
lam=[];expected=names;if c.reactive,lam=dataset(path,'/state/lambda',n,'double')';expected{end+1}='lambda';end
info=h5info(path,'/state');assert(isequal(sort({info.Datasets.Name}),sort(expected)),'Unexpected adaptive state fields');
s=struct('config',c,'state',U,'lam',lam,'beta',[],'mesh',M,'workers',1);
for key={'time','steps','regrids','mesh_revision','budget_limited','run_id'},s.(key{1})=m.(key{1});end
for key={'initial_integral','boundary_integral','source_integral'},s.(key{1})=m.(key{1})(:);end
torchAdaptiveData(s);current=torchSessionIntegral(s);saved=m.current_integral(:);
assert(numel(saved)==d+2&&all(isfinite(saved))&&all(abs(saved-current)<=1e-12+1e-12*abs(current)),'State/integral mismatch');
end
function value=dataset(path,key,n,type)
info=h5info(path,key);assert(numel(info.Dataspace.Size)==1&&info.Dataspace.Size==n,'Invalid adaptive dataset rank');
value=h5read(path,key);assert(isa(value,type)&&numel(value)==n&&all(isfinite(value(:))),'Invalid adaptive dataset precision/values');value=value(:);
end
