function out = torchCartesianVelocity(c,P)
% P is canonical [components,x-fastest] primitives. Return 3 Cartesian rows.
d=c.dimension;g=c.geometry;out=zeros(3,size(P,2));out(1:d,:)=P(2:end-1,:);
if strcmp(g,'cartesian')||d==1,return;end
if strcmp(g,'cylindrical')&&d==2,out(3,:)=out(2,:);out(2,:)=0;return;end
G=torchSessionGrid(c);a=cell(1,3);for k=1:3,if k<=d,a{k}=(G.axes{k}(1:end-1)+G.axes{k}(2:end))/2;else,a{k}=0;end;end
[~,T,F]=ndgrid(a{1},a{2},a{3});t=T(:)';f=F(:)';v=out;
if strcmp(g,'spherical')
    out(1,:)=(v(1,:).*sin(t)+v(2,:).*cos(t)).*cos(f)-v(3,:).*sin(f);
    out(2,:)=(v(1,:).*sin(t)+v(2,:).*cos(t)).*sin(f)+v(3,:).*cos(f);
    out(3,:)=v(1,:).*cos(t)-v(2,:).*sin(t);
else,out(1,:)=v(1,:).*cos(t)-v(2,:).*sin(t);out(2,:)=v(1,:).*sin(t)+v(2,:).*cos(t);end
end
