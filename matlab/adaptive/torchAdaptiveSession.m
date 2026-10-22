function s = torchAdaptiveSession(s,options)
% Convert a uniform session to conservative dyadic AMR, preserving balances.
if nargin<2,options=struct;end
assert(~isfield(s,'mesh'),'Session is already adaptive');c=s.config;c.amr=torchAdaptiveOptions(options);
faces=[];if isfield(c,'face_boundaries'),faces=c.face_boundaries;end;c.face_boundaries=torchBoundaryFaces(c,faces);c=torchAdaptiveValidate(c);
U=torchGridOrder(s.state,c,'flatten');Q=[];
if c.reactive,shape=size(s.state);Q=torchGridOrder(reshape(s.lam,[1,shape(2:end)]),c,'flatten');end
s.config=c;s.state=U;s.lam=Q;s.beta=[];s.mesh=torchAdaptiveMesh(c);s.regrids=0;s.mesh_revision=0;s.budget_limited=false;
if ~isfield(s,'workers'),s.workers=1;end
for level=1:c.amr.max_level,s=torchAdaptiveRegrid(s);end
end
