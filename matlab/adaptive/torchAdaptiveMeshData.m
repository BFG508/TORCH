function [points,cells]=torchAdaptiveMeshData(s)
% Each leaf owns its vertices; connectivity uses zero-based VTK/XDMF indices.
d=s.config.dimension;G=s.mesh;n=size(G.keys,1);
offsets={[0;1],[0,0;1,0;1,1;0,1],[0,0,0;1,0,0;1,1,0;0,1,0;0,0,1;1,0,1;1,1,1;0,1,1]};
o=offsets{d};nodes=size(o,1);points=zeros(n*nodes,3);
for k=1:nodes,points(k:nodes:end,1:d)=G.low+o(k,:).*G.widths;end
points=torchMapPoints(s.config,points);cells=reshape(int64(0:n*nodes-1),nodes,n)';
end
