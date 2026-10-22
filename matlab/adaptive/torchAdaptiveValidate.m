function c = torchAdaptiveValidate(c)
% Validate non-executable configuration independently of a leaf state layout.
d=c.dimension;assert(isscalar(d)&&any(d==1:3),'Invalid dimension');c.amr=torchAdaptiveOptions(c.amr);
for key={'counts_xyz','lengths_xyz','origins_xyz'},value=c.(key{1});assert(isnumeric(value)&&numel(value)==d&&all(isfinite(value)),'Invalid adaptive grid metadata');c.(key{1})=value(:)';end
assert(all(c.counts_xyz>=2&c.counts_xyz==fix(c.counts_xyz))&&all(c.lengths_xyz>0),'Invalid adaptive base mesh');
assert(isscalar(c.gamma)&&isfinite(c.gamma)&&c.gamma>1&&isscalar(c.cfl)&&isfinite(c.cfl)&&c.cfl>0&&c.cfl<=.5,'Invalid gamma/CFL');
assert(isscalar(c.t_end)&&isfinite(c.t_end)&&c.t_end>=0&&isscalar(c.max_steps)&&isfinite(c.max_steps)&&c.max_steps>=1&&c.max_steps==fix(c.max_steps),'Invalid end time/step limit');
assert(isscalar(c.reactive)&&any(c.reactive==[0,1])&&strcmp(c.kinetic_model,'arrhenius'),'MATLAB adaptive sessions support inert/Arrhenius; ZND is Python-only');
assert(strcmp(c.controller,'Sound Speed Controlled')&&strcmp(c.variant,'Variante 1 - Original')&&strcmp(c.wavespeed,'Pressure Based'),'AMR uses its own limited-linear Rusanov kernel');
assert(any(strcmp(c.limiter,{'VanLeer','Albada','MINBEE','Superbee','Unos'})),'Invalid base limiter');
if ischar(c.boundaries_xyz),c.boundaries_xyz={c.boundaries_xyz};end;c.boundaries_xyz=c.boundaries_xyz(:)';assert(numel(c.boundaries_xyz)==d,'Invalid base boundaries');
for k=1:d,assert(any(strcmp(c.boundaries_xyz{k},{'Open','Reflective','Periodic'}))||(d==2&&k==2&&strcmp(c.boundaries_xyz{k},'Inflow')),'Invalid base boundary');end
c.face_boundaries=torchBoundaryFaces(c,c.face_boundaries);torchMetricGrid(c);
end
