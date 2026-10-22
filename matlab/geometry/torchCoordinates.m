function names = torchCoordinates(c)
% Logical coordinate names; velocity/momentum components use the local basis.
g='cartesian';if isfield(c,'geometry'),g=c.geometry;end;d=c.dimension;
if strcmp(g,'cartesian'),names={'x','y','z'};
elseif strcmp(g,'spherical'),names={'r','theta','phi'};
elseif strcmp(g,'cylindrical') && d==2,names={'r','z'};
else,names={'r','theta','z'};end
names=names(1:d);
end
