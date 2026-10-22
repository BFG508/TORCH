function varargout = torchMetricStep(action,s,G,dt)
% Metric MUSCL/Rusanov finite volumes with SSPRK2 and positivity fallback.
assert(any(strcmp(action,{'dt','rhs','advance'})),'Unknown metric operation');
c=s.config;d=c.dimension;U=torchGridOrder(s.state,c,'flatten');Q=zeros(0,size(U,2));
if c.reactive,shape=size(s.state);Q=torchGridOrder(reshape(s.lam,[1,shape(2:end)]),c,'flatten');end
if strcmp(action,'rhs')
    [du,dq,b,source]=rhs(U,Q,c,G,false);varargout={du,dq,b,source};return;
end
if strcmp(action,'dt')
    g=gas(Q,c,size(U,2));P=primitive(U,g);rate=zeros(1,size(U,2));
    for k=1:d
        [L,R,gl,gr]=faces(P,g,Q,c,G,k);a=max(abs(L(k+1,:))+sqrt(gl.*L(end,:)./L(1,:)),abs(R(k+1,:))+sqrt(gr.*R(end,:)./R(1,:)));
        value=a.*G.area{k};ix=G.index{k};rate=rate+max(value(ix.cell_low),value(ix.cell_high))./G.volume;
    end
    varargout={c.cfl/max(rate)};return;
end
for first=[false,true]
    rhoq=Q.*U(1,:);[k1,q1,b1,s1]=rhs(U,Q,c,G,first);stage=U+dt*k1;
    if ~physical(stage),continue;end
    stageq=(rhoq+dt*q1)./stage(1,:);if any(stageq(:)<-1e-12 | stageq(:)>1+1e-12),continue;end;stageq=max(0,min(1,stageq));
    [k2,q2,b2,s2]=rhs(stage,stageq,c,G,first);out=.5*(U+stage+dt*k2);
    if ~physical(out),continue;end
    finalq=.5*(rhoq+stage(1,:).*stageq+dt*q2)./out(1,:);
    if any(finalq(:)<-1e-12 | finalq(:)>1+1e-12),continue;end;finalq=max(0,min(1,finalq));
    lam=[];if c.reactive,shape=size(s.lam);value=torchGridOrder(finalq,c,'restore');lam=reshape(value,shape);end
    varargout={torchGridOrder(out,c,'restore'),lam,.5*dt*(b1+b2),.5*dt*(s1+s2)};return;
end
error('TORCH:MetricPositivity','Metric/face step lost positivity');
end

function [change,species,boundary,integrated] = rhs(U,Q,c,G,first)
d=c.dimension;g=gas(Q,c,size(U,2));P=primitive(U,g);source=zeros(size(U));geometry=c.geometry;
if ~strcmp(geometry,'cartesian')
    rho=P(1,:);p=P(end,:);radial=G.divergence{1};source(2,:)=p.*radial;
    if strcmp(geometry,'spherical')
        invr=radial/2;source(2,:)=source(2,:)+rho.*sum(P(3:end-1,:).^2,1).*invr;
        if d>=2
            cotr=G.divergence{2};source(3,:)=p.*cotr-rho.*P(2,:).*P(3,:).*invr;
            if d==3,source(3,:)=source(3,:)+rho.*P(4,:).^2.*cotr;source(4,:)=-rho.*P(4,:).*(P(2,:).*invr+P(3,:).*cotr);end
        end
    elseif d>=2 && ~(strcmp(geometry,'cylindrical') && d==2)
        source(2,:)=source(2,:)+rho.*P(3,:).^2.*radial;source(3,:)=-rho.*P(2,:).*P(3,:).*radial;
    end
end
change=source;species=zeros(size(Q));boundary=zeros(d+2,1);
for k=1:d
    ix=G.index{k};[L,R,gl,gr,ql,qr]=faces(P,g,Q,c,G,k);
    if ~first
        before=P(:,ix.minus);after=P(:,ix.plus);
        if ~strcmp(c.face_boundaries{2*k-1}.type,'periodic')
            [before(:,ix.at_low),~,~]=torchBoundaryGhost(P(:,ix.at_low),g(ix.at_low),Q(:,ix.at_low),c.face_boundaries{2*k-1},k,-1,c.reactive);
            [after(:,ix.at_high),~,~]=torchBoundaryGhost(P(:,ix.at_high),g(ix.at_high),Q(:,ix.at_high),c.face_boundaries{2*k},k,1,c.reactive);
        end
        slope=limited(P-before,after-P,c.limiter);invalid=any(P([1,end],:)-.5*abs(slope([1,end],:))<=1e-12,1);slope(:,invalid)=0;
        L=L+.5*slope(:,ix.left);R=R-.5*slope(:,ix.right);
        if ~strcmp(c.face_boundaries{2*k-1}.type,'periodic'),L(:,ix.low)=facesGhost(P,g,Q,c,G,k,-1);R(:,ix.high)=facesGhost(P,g,Q,c,G,k,1);end
    end
    if strcmp(c.face_boundaries{2*k-1}.type,'wall'),L(:,ix.low)=R(:,ix.low);L(k+1,ix.low)=-L(k+1,ix.low);gl(ix.low)=gr(ix.low);end
    if strcmp(c.face_boundaries{2*k}.type,'wall'),R(:,ix.high)=L(:,ix.high);R(k+1,ix.high)=-R(k+1,ix.high);gr(ix.high)=gl(ix.high);end
    ul=conservative(L,gl);ur=conservative(R,gr);fl=ul.*L(k+1,:);fr=ur.*R(k+1,:);
    fl(k+1,:)=fl(k+1,:)+L(end,:);fr(k+1,:)=fr(k+1,:)+R(end,:);
    fl(end,:)=fl(end,:)+L(k+1,:).*L(end,:);fr(end,:)=fr(end,:)+R(k+1,:).*R(end,:);
    speed=max(abs(L(k+1,:))+sqrt(gl.*L(end,:)./L(1,:)),abs(R(k+1,:))+sqrt(gr.*R(end,:)./R(1,:)));
    flux=.5*(fl+fr-speed.*(ur-ul));weighted=flux.*G.area{k};
    change=change-(weighted(:,ix.cell_high)-weighted(:,ix.cell_low))./G.volume;
    boundary=boundary+sum(weighted(:,ix.high),2)-sum(weighted(:,ix.low),2);
    upwind=ql;mask=flux(1,:)<0;upwind(:,mask)=qr(:,mask);transport=flux(1,:).*upwind.*G.area{k};
    species=species-(transport(:,ix.cell_high)-transport(:,ix.cell_low))./G.volume;
end
integrated=sum(source.*G.volume,2);
end

function [L,R,gl,gr,ql,qr] = faces(P,g,Q,c,G,k)
ix=G.index{k};L=P(:,ix.left);R=P(:,ix.right);gl=g(ix.left);gr=g(ix.right);ql=Q(:,ix.left);qr=Q(:,ix.right);
if ~strcmp(c.face_boundaries{2*k-1}.type,'periodic')
    [L(:,ix.low),gl(ix.low),ql(:,ix.low)]=torchBoundaryGhost(L(:,ix.low),gl(ix.low),ql(:,ix.low),c.face_boundaries{2*k-1},k,-1,c.reactive);
    [R(:,ix.high),gr(ix.high),qr(:,ix.high)]=torchBoundaryGhost(R(:,ix.high),gr(ix.high),qr(:,ix.high),c.face_boundaries{2*k},k,1,c.reactive);
end
end

function value=facesGhost(P,g,Q,c,G,k,side)
ix=G.index{k};mask=ix.low;indices=ix.left;f=c.face_boundaries{2*k-1};
if side==1,mask=ix.high;indices=ix.right;f=c.face_boundaries{2*k};end
[value,~,~]=torchBoundaryGhost(P(:,indices(mask)),g(indices(mask)),Q(:,indices(mask)),f,k,side,c.reactive);
end

function g=gas(Q,c,n)
g=c.gamma*ones(1,n);if c.reactive,g=(4.5-Q(1,:))./(3.5-Q(1,:));end
end
function P=primitive(U,g)
P=U;P(2:end-1,:)=U(2:end-1,:)./U(1,:);P(end,:)=(g-1).*(U(end,:)-.5*sum(U(2:end-1,:).^2,1)./U(1,:));
end
function U=conservative(P,g)
U=P;U(2:end-1,:)=P(2:end-1,:).*P(1,:);U(end,:)=P(end,:)./(g-1)+.5*P(1,:).*sum(P(2:end-1,:).^2,1);
end
function yes=physical(U)
yes=all(isfinite(U(:))) && all(U(1,:)>1e-12) && all(U(end,:)-.5*sum(U(2:end-1,:).^2,1)./U(1,:)>1e-12);
end
function slope=limited(minus,plus,name)
den=plus;den(abs(den)<1e-14)=1e-14;r=minus./den;pos=max(r,0);
switch name
case 'VanLeer',phi=2*pos./(1+pos);
case 'Albada',phi=(pos.^2+pos)./(pos.^2+1);
case 'MINBEE',phi=max(0,min(r,1));
case 'Superbee',phi=max(0,max(min(2*r,1),min(r,2)));
otherwise,phi=max(0,min(min(2,2*r),min((2+r)/3,(1+2*r)/3)));
end
slope=phi.*plus;
end
