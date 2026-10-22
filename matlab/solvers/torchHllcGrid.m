function flux = torchHllcGrid(left,right,gamma_left,gamma_right,direction,batch_size)
% Directional grid flux with HLLC scratch arrays bounded by the batch size.
% Inputs/output keep Cartesian component order. Rotation is batch-local.
if nargin<5,direction=1;end
if nargin<6,batch_size=8192;end
nv=size(left,1);shape=size(left);n=numel(left)/nv;
assert(isequal(size(left),size(right))&&any(nv==[4,5]),'Invalid HLLC grid states');
assert(isscalar(direction)&&any(direction==1:nv-2),'Invalid normal direction');
assert(isscalar(batch_size)&&isfinite(batch_size)&&batch_size>=1&&batch_size==fix(batch_size),'Invalid HLLC batch size');
L=reshape(left,nv,[]);R=reshape(right,nv,[]);
if isscalar(gamma_left),gl=repmat(gamma_left,1,n);else,gl=reshape(gamma_left,1,[]);end
if isscalar(gamma_right),gr=repmat(gamma_right,1,n);else,gr=reshape(gamma_right,1,[]);end
assert(numel(gl)==n&&numel(gr)==n,'Gamma must match the grid faces');
normal=direction+1;order=[1,normal,setdiff(2:nv-1,normal),nv];F=zeros(nv,n);
for start=1:batch_size:n
    ix=start:min(n,start+batch_size-1);
    F(order,ix)=torchHllcFlux(L(order,ix),R(order,ix),gl(ix),gr(ix));
end
flux=reshape(F,shape);
end
