function session = torchCreateSession(P, lengths, varargin)
% Bounded-memory session. Primitive dimensions: [3,nx], [4,ny,nx], [5,nz,ny,nx].
% Config, checkpoints and spatial conventions match Python Simulation.
dim=size(P,1)-2; assert(any(dim==[1,2,3]),'Invalid primitive components');
shape=size(P); counts=shape(end:-1:2);
assert(numel(counts)==dim && numel(lengths)==dim,'Mesh dimension mismatch');
p=inputParser;
addParameter(p,'t_end',.2); addParameter(p,'cfl',.4); addParameter(p,'gamma',1.4);
addParameter(p,'boundaries',repmat({'Open'},1,dim)); addParameter(p,'limiter','VanLeer');
addParameter(p,'reactive',false); addParameter(p,'lambda_init',[]);
addParameter(p,'inlet_primitive',[1,0,100,101325]); addParameter(p,'controller','Sound Speed Controlled');
addParameter(p,'variant','Variante 1 - Original'); addParameter(p,'wavespeed','Pressure Based');
addParameter(p,'kinetic_model','arrhenius'); addParameter(p,'max_steps',100000); parse(p,varargin{:});o=p.Results;
c=rmfield(o,{'boundaries','lambda_init'}); c.dimension=dim; c.counts_xyz=counts;
c.lengths_xyz=lengths(:)'; c.boundaries_xyz=o.boundaries;
assert(any(strcmp(c.kinetic_model,{'arrhenius','znd'})),'Unknown kinetic model');
assert(~c.reactive || strcmp(c.kinetic_model,'arrhenius'),'MATLAB sessions support Arrhenius; ZND remains Python-only');
assert(all(isfinite([counts,lengths(:)',c.t_end,c.cfl,c.gamma])) && all(counts>=2) && all(lengths>0) && c.t_end>=0 && c.cfl>0 && c.cfl<=.5 && c.gamma>1,'Invalid mesh, gamma, CFL or time');
assert(iscell(c.boundaries_xyz) && numel(c.boundaries_xyz)==dim,'Invalid boundaries');
for k=1:dim
    valid=any(strcmp(c.boundaries_xyz{k},{'Open','Reflective','Periodic'})) || (dim==2 && k==2 && strcmp(c.boundaries_xyz{k},'Inflow'));
    assert(valid,'Unsupported boundary');
end
assert(any(strcmp(c.limiter,{'VanLeer','Albada','MINBEE','Superbee','Unos'})),'Invalid limiter');
assert(any(strcmp(c.controller,{'Sound Speed Controlled','w=0'})),'Invalid controller');
assert(any(strcmp(c.variant,{'Variante 1 - Original','Variante_2'})),'Invalid HLLC variant');
assert(any(strcmp(c.wavespeed,{'Pressure Based','Enthalpy Based'})),'Invalid wavespeed');
assert(isscalar(c.max_steps) && isfinite(c.max_steps) && c.max_steps>=1 && c.max_steps==fix(c.max_steps),'Invalid maximum steps');
P=double(P); positive=P([1,end],:);
assert(all(isfinite(P(:))) && all(positive(:)>0),'Nonphysical primitives');
lam=[]; gamma=c.gamma;
if c.reactive
    lam=o.lambda_init; if isempty(lam), if dim==1,lam=ones(1,counts);else,lam=ones(shape(2:end));end;end
    expected=shape(2:end);if dim==1,expected=[1,counts];end
    assert(isequal(size(lam),expected) && all(isfinite(lam(:))) && all(lam(:)>=0 & lam(:)<=1),'Invalid lambda');
    if dim==1, lam=reshape(lam,1,[]); else, lam=reshape(lam,shape(2:end)); end
    gamma=(4.5-lam)./(3.5-lam);
else, assert(isempty(o.lambda_init),'Composition requires reactive chemistry'); end
flat=reshape(P,dim+2,[]); g=gamma; if ~isscalar(g), g=gamma(:)'; end
U=flat; U(2:end-1,:)=flat(2:end-1,:).*flat(1,:);
U(end,:)=flat(end,:)./(g-1)+.5*flat(1,:).*sum(flat(2:end-1,:).^2,1); U=reshape(U,shape);
check=reshape(U,dim+2,[]);internal=check(end,:)-.5*sum(check(2:end-1,:).^2,1)./check(1,:);
assert(all(isfinite(check(:))) && all(check(1,:)>1e-12) && all(internal>1e-12),'Nonphysical conservative state');
h=lengths(:)'./counts;
[~,identity]=fileparts(tempname);
session=struct('config',c,'state',U,'lam',lam,'beta',[],'time',0,'steps',0,'run_id',identity,...
    'initial_integral',sum(reshape(U,dim+2,[]),2)*prod(h),'boundary_integral',zeros(dim+2,1),'source_integral',zeros(dim+2,1));
% Reject unsupported inputs before stepping or writing a checkpoint.
if dim==1, torchSolver('session_dt',U,gamma,h,c);
elseif dim==2, torchSolver2D('session_dt',U,gamma,h,c);
else, torchSolver3D('session_dt',U,gamma,h,c); end
end
