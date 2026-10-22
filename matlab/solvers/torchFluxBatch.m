function F = torchFluxBatch(kind,L,R,gl,gr,direction)
% Pure worker function: no GUI, files, global reductions or mutable sessions.
if strcmp(kind,'hllc')
    nv=size(L,1);normal=direction+1;order=[1,normal,setdiff(2:nv-1,normal),nv];
    value=torchHllcFlux(L(order,:),R(order,:),gl,gr);F=zeros(size(L));F(order,:)=value;return;
end
assert(strcmp(kind,'rusanov'),'Unknown flux batch');
ul=L;ur=R;ul(2:end-1,:)=L(2:end-1,:).*L(1,:);ur(2:end-1,:)=R(2:end-1,:).*R(1,:);
ul(end,:)=L(end,:)./(gl-1)+.5*L(1,:).*sum(L(2:end-1,:).^2,1);
ur(end,:)=R(end,:)./(gr-1)+.5*R(1,:).*sum(R(2:end-1,:).^2,1);
fl=ul.*L(direction+1,:);fr=ur.*R(direction+1,:);
fl(direction+1,:)=fl(direction+1,:)+L(end,:);fr(direction+1,:)=fr(direction+1,:)+R(end,:);
fl(end,:)=fl(end,:)+L(direction+1,:).*L(end,:);fr(end,:)=fr(end,:)+R(direction+1,:).*R(end,:);
speed=max(abs(L(direction+1,:))+sqrt(gl.*L(end,:)./L(1,:)),abs(R(direction+1,:))+sqrt(gr.*R(end,:)./R(1,:)));
F=.5*(fl+fr-speed.*(ur-ul));
end
