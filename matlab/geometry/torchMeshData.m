function [points,cells] = torchMeshData(c)
% Physical Cartesian vertices and x-fastest line/quad/hexa connectivity.
d=c.dimension;n=[c.counts_xyz,ones(1,3-d)];lengths=[c.lengths_xyz,zeros(1,3-d)];o=zeros(1,3);
if isfield(c,'origins_xyz'),o(1:d)=c.origins_xyz;end
a=cell(1,3);for k=1:3,if k<=d,a{k}=linspace(o(k),o(k)+lengths(k),n(k)+1);else,a{k}=0;end;end
[X,Y,Z]=ndgrid(a{1},a{2},a{3});points=torchMapPoints(c,[X(:),Y(:),Z(:)]);
[I,J,K]=ndgrid(0:n(1)-1,0:n(2)-1,0:n(3)-1);
nyNodes=1;if d>=2,nyNodes=n(2)+1;end
base=I+(n(1)+1)*(J+nyNodes*K);
if d==1,offsets=[0,1];elseif d==2,offsets=[0,1,n(1)+2,n(1)+1];
else,plane=(n(1)+1)*(n(2)+1);offsets=[0,1,n(1)+2,n(1)+1,plane,plane+1,plane+n(1)+2,plane+n(1)+1];end
cells=base(:)+offsets;
end
