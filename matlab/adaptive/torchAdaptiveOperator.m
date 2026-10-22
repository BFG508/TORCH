function varargout = torchAdaptiveOperator(action,s,dt)
% Global-step MUSCL/SSPRK2 on conservative shared coarse/fine subfaces.
M=s.mesh;U=s.state;c=s.config;Q=zeros(0,size(U,2));if c.reactive,Q=s.lam;end
workers=1;if isfield(s,'workers'),workers=s.workers;end
if strcmp(action,'dt')
    [L,R,gl,gr]=torchAdaptiveFaces(M,U,Q,c,true);rate=zeros(size(U,2),1);
    for k=1:c.dimension
        mask=M.directions==k;speed=max(abs(L(k+1,mask))+sqrt(gl(mask).*L(end,mask)./L(1,mask)),abs(R(k+1,mask))+sqrt(gr(mask).*R(end,mask)./R(1,mask)));
        value=(speed.*M.areas(mask))';
        for side=1:2,ids=M.left(mask);if side==2,ids=M.right(mask);end;active=ids>0;rate=rate+accumarray(ids(active),value(active),[size(U,2),1]);end
    end
    varargout={c.cfl/max(.5*rate'./M.volumes)};return;
elseif strcmp(action,'rhs')
    [du,dq,b,source]=rhs(M,U,Q,c,workers,false);varargout={du,dq,b,source};return;
end
assert(strcmp(action,'advance'),'Unknown adaptive operation');
rhoq=U(1,:).*Q;
for first=[false,true]
    [k1,q1,b1,s1]=rhs(M,U,Q,c,workers,first);stage=U+dt*k1;if ~torchAdaptivePhysical(stage),continue;end
    stageq=(rhoq+dt*q1)./stage(1,:);if any(~isfinite(stageq(:))|stageq(:)<-1e-12|stageq(:)>1+1e-12),continue;end;stageq=max(0,min(1,stageq));
    [k2,q2,b2,s2]=rhs(M,stage,stageq,c,workers,first);out=.5*(U+stage+dt*k2);if ~torchAdaptivePhysical(out),continue;end
    finalq=.5*(rhoq+stage(1,:).*stageq+dt*q2)./out(1,:);if any(~isfinite(finalq(:))|finalq(:)<-1e-12|finalq(:)>1+1e-12),continue;end
    varargout={out,max(0,min(1,finalq)),.5*dt*(b1+b2),.5*dt*(s1+s2)};return;
end
error('TORCH:AdaptivePositivity','Adaptive update lost positivity');
end

function [change,species,boundary,integrated]=rhs(M,U,Q,c,workers,first)
[L,R,gl,gr,ql,qr]=torchAdaptiveFaces(M,U,Q,c,~first);F=zeros(size(L));N=size(U,2);
for k=1:c.dimension,mask=M.directions==k;F(:,mask)=torchParallelFlux('rusanov',L(:,mask),R(:,mask),gl(mask),gr(mask),k,workers);end
P=torchAdaptivePrimitive(U,Q,c);source=torchAdaptiveSource(M,P,c);change=source.*M.volumes;species=zeros(size(Q));boundary=zeros(size(U,1),1);weighted=F.*M.areas;
for side=1:2
    ids=M.left;sign=-1;outward=1;if side==2,ids=M.right;sign=1;outward=-1;end
    mask=ids>0;
    for v=1:size(U,1),change(v,:)=change(v,:)+sign*accumarray(ids(mask),weighted(v,mask)',[N,1])';end
    boundary=boundary+outward*sum(weighted(:,M.sides==outward),2);
    for q=1:size(Q,1)
        upwind=ql(q,:);negative=F(1,:)<0;upwind(negative)=qr(q,negative);transport=weighted(1,:).*upwind;
        species(q,:)=species(q,:)+sign*accumarray(ids(mask),transport(mask)',[N,1])';
    end
end
change=change./M.volumes;species=species./M.volumes;integrated=sum(source.*M.volumes,2);
end
