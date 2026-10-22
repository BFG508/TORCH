function G = torchSessionGrid(c)
% Reuse one metric grid between session calls; MATLAB arrays use copy-on-write.
% The key includes all geometry/index dependencies, including periodic axes.
% Only one grid is retained. Use "clear torchSessionGrid" to release it.
persistent previous grid
d=c.dimension;geometry='cartesian';if isfield(c,'geometry'),geometry=c.geometry;end
origins=zeros(1,d);if isfield(c,'origins_xyz'),origins=c.origins_xyz;end
periodic=false(1,d);
if isfield(c,'face_boundaries')
    for k=1:d,periodic(k)=strcmp(c.face_boundaries{2*k-1}.type,'periodic');end
end
key={d,geometry,c.counts_xyz(:)',c.lengths_xyz(:)',origins(:)',periodic};
if isempty(grid)||~isequal(previous,key)
    candidate=torchMetricGrid(c);grid=candidate;previous=key;
end
G=grid;
end
