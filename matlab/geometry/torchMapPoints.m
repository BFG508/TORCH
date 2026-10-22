function out = torchMapPoints(c,P)
% Logical coordinate triples to physical Cartesian positions.
out=P;g='cartesian';if isfield(c,'geometry'),g=c.geometry;end;d=c.dimension;
if strcmp(g,'cartesian')||d==1,return;end
r=P(:,1);theta=P(:,2);
if strcmp(g,'spherical')
    phi=zeros(size(r));if d==3,phi=P(:,3);end
    out=[r.*sin(theta).*cos(phi),r.*sin(theta).*sin(phi),r.*cos(theta)];
elseif strcmp(g,'cylindrical')&&d==2,out=[r,zeros(size(r)),theta];
else,out(:,1)=r.*cos(theta);out(:,2)=r.*sin(theta);end
end
