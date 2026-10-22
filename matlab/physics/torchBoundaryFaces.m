function faces = torchBoundaryFaces(c,specification)
% Per-face dictionaries, ordered x_min,x_max,y_min,y_max,z_min,z_max.
d=c.dimension;faces=cell(1,2*d);
for k=1:d
    bc=c.boundaries_xyz{k};kind='outflow';
    if strcmp(bc,'Reflective'),kind='wall';elseif strcmp(bc,'Periodic'),kind='periodic';elseif strcmp(bc,'Inflow'),kind='inflow';end
    f=struct('type',kind);if strcmp(kind,'inflow'),f.primitive=c.inlet_primitive;end
    faces{2*k-1}=f;faces{2*k}=f;if strcmp(kind,'inflow'),faces{2*k}=struct('type','outflow');end
end
if nargin>=2 && ~isempty(specification)
    if isstruct(specification) && isscalar(specification) && ~isfield(specification,'type')
        names={'x_min','x_max','y_min','y_max','z_min','z_max'};
        keys=fieldnames(specification);assert(all(ismember(keys,names(1:2*d))),'Unknown face key');
        for k=1:2*d,if isfield(specification,names{k}),faces{k}=specification.(names{k});end;end
    else
        if iscell(specification),faces=specification(:)';else,faces=num2cell(specification(:)');end
    end
end
assert(numel(faces)==2*d,'One boundary per face required');
for k=1:2*d
    f=faces{k};if ischar(f),f=struct('type',f);end
    assert(isstruct(f) && isfield(f,'type') && any(strcmp(f.type,{'outflow','wall','periodic','inflow','pressure_outlet','stagnation_inlet'})),'Unknown face boundary');
    allowed={'type','lambda','beta'};
    if strcmp(f.type,'inflow'),allowed=[allowed,{'primitive'}];assert(isfield(f,'primitive'),'Inflow requires primitives');end
    if strcmp(f.type,'pressure_outlet'),allowed=[allowed,{'pressure','backflow'}];positive={'pressure'};
    elseif strcmp(f.type,'stagnation_inlet'),allowed=[allowed,{'total_pressure','total_temperature','gas_constant'}];positive={'total_pressure','total_temperature','gas_constant'};if ~isfield(f,'gas_constant'),f.gas_constant=287;end
    else,positive={};end
    assert(all(ismember(fieldnames(f),allowed)),'Unknown boundary parameter');
    for name=positive,value=f.(name{1});assert(isscalar(value)&&isfinite(value)&&value>0,'Positive boundary parameter required');end
    for name={'lambda','beta'}
        if isfield(f,name{1}),value=f.(name{1});assert(isscalar(value)&&isfinite(value)&&value>=0&&value<=1,'Invalid inlet fraction');end
    end
    for name={'primitive','backflow'}
        if isfield(f,name{1})
            P=f.(name{1});P=P(:)';sign=1;if mod(k,2)==1,sign=-1;end
            assert(numel(P)==d+2&&all(isfinite(P))&&P(1)>0&&P(end)>0&&sign*P(ceil(k/2)+1)<0,'Invalid inlet/backflow primitives');f.(name{1})=P;
        end
    end
    faces{k}=f;
end
for k=1:d,assert(strcmp(faces{2*k-1}.type,'periodic')==strcmp(faces{2*k}.type,'periodic'),'Periodic boundaries must be paired');end
g='cartesian';if isfield(c,'geometry'),g=c.geometry;end
o=zeros(1,d);if isfield(c,'origins_xyz'),o=c.origins_xyz;end
if ~strcmp(g,'cartesian')
    assert(~any(cellfun(@(f)strcmp(f.type,'periodic'),faces(1:2))),'Radius cannot be periodic');
    if o(1)==0,assert(strcmp(faces{1}.type,'wall'),'Radial origin requires wall');end
    if strcmp(g,'spherical') && d>=2
        if o(2)==0,assert(strcmp(faces{3}.type,'wall'),'Spherical pole requires wall');end
        if abs(o(2)+c.lengths_xyz(2)-pi)<1e-14,assert(strcmp(faces{4}.type,'wall'),'Spherical pole requires wall');end
        assert(~any(cellfun(@(f)strcmp(f.type,'periodic'),faces(3:4))),'Polar theta cannot be periodic');
    end
    az=0;if strcmp(g,'spherical')&&d==3,az=3;elseif (strcmp(g,'annular')&&d>=2)||(strcmp(g,'cylindrical')&&d==3),az=2;end
    if az>0 && strcmp(faces{2*az-1}.type,'periodic'),assert(abs(c.lengths_xyz(az)-2*pi)<1e-14,'Azimuthal periodicity requires 2*pi');end
end
end
