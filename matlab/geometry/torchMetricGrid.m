function G = torchMetricGrid(c)
% Exact volumes/face areas on uniform Cartesian and orthogonal curved grids.
d=c.dimension;g='cartesian';if isfield(c,'geometry'),g=c.geometry;end
assert(any(strcmp(g,{'cartesian','cylindrical','annular','spherical'})),'Unknown geometry');
n=c.counts_xyz;L=c.lengths_xyz;o=zeros(1,d);if isfield(c,'origins_xyz'),o=c.origins_xyz;end
assert(numel(o)==d && all(isfinite(o)),'Invalid origins');
a=cell(1,d);w=a;
for k=1:d,a{k}=linspace(o(k),o(k)+L(k),n(k)+1);w{k}=diff(a{k});end
if ~strcmp(g,'cartesian')
    assert(o(1)>=0 && (~strcmp(g,'annular') || o(1)>0),'Invalid inner radius');
    if strcmp(g,'cylindrical') && d==3,assert(o(1)>0,'3D cylindrical grids require positive inner radius');end
    if strcmp(g,'spherical') && d>=2
        assert(o(2)>=0 && o(2)+L(2)<=pi+1e-14,'Spherical theta outside [0,pi]');
        if d==3,assert(o(1)>0 && o(2)>0 && o(2)+L(2)<pi,'3D spherical grids require r>0 and 0<theta<pi');end
    end
    az=0;if strcmp(g,'spherical') && d==3,az=3;elseif (strcmp(g,'annular') && d>=2)||(strcmp(g,'cylindrical') && d==3),az=2;end
    if az>0,assert(L(az)<=2*pi+1e-14 && L(az)/n(az)<pi,'Invalid azimuthal extent/cell angle');end
end
F=cell(1,d);
if strcmp(g,'cartesian')
    V=w;
    for k=1:d,F{k}=w;F{k}{k}=ones(1,n(k)+1);end
else
    r=a{1};dr=diff(r);s2=diff(r.^2)/2;
    if strcmp(g,'spherical')
        s3=diff(r.^3)/3;
        if d==1,V={4*pi*s3};F{1}={4*pi*r.^2};
        else
            t=a{2};solid=-diff(cos(t));factor=1;if d==2,factor=2*pi;end
            V={factor*s3,solid};F{1}={factor*r.^2,solid};F{2}={factor*s2,sin(t)};
            if d==3,V{3}=w{3};F{1}{3}=w{3};F{2}{3}=w{3};F{3}={s2,w{2},ones(1,n(3)+1)};end
        end
    elseif d==1,V={2*pi*s2};F{1}={2*pi*r};
    elseif strcmp(g,'cylindrical') && d==2
        V={2*pi*s2,w{2}};F{1}={2*pi*r,w{2}};F{2}={2*pi*s2,ones(1,n(2)+1)};
    else
        V={s2,w{2}};F{1}={r,w{2}};F{2}={dr,ones(1,n(2)+1)};
        if d==3,V{3}=w{3};F{1}{3}=w{3};F{2}{3}=w{3};F{3}={s2,w{2},ones(1,n(3)+1)};end
    end
end
G=struct('axes',{a},'volume',product(V),'area',{{}},'divergence',{{}},'index',{{}});
assert(all(isfinite(G.volume)) && all(G.volume>0),'Invalid volumes');
cellcoord=coords(n);stride=[1,cumprod(n(1:end-1))];
for k=1:d
    facecount=n;facecount(k)=n(k)+1;facecoord=coords(facecount);
    left=facecoord;right=facecoord;left(k,:)=left(k,:)-1;
    periodic=false;if isfield(c,'face_boundaries'),periodic=strcmp(c.face_boundaries{2*k-1}.type,'periodic');end
    if periodic,left(k,:)=mod(left(k,:),n(k));right(k,:)=mod(right(k,:),n(k));
    else,left(k,:)=max(0,min(n(k)-1,left(k,:)));right(k,:)=max(0,min(n(k)-1,right(k,:)));end
    fstride=[1,cumprod(facecount(1:end-1))];
    low=cellcoord;high=cellcoord;high(k,:)=high(k,:)+1;
    minus=cellcoord;plus=cellcoord;minus(k,:)=minus(k,:)-1;plus(k,:)=plus(k,:)+1;
    if periodic,minus(k,:)=mod(minus(k,:),n(k));plus(k,:)=mod(plus(k,:),n(k));
    else,minus(k,:)=max(0,minus(k,:));plus(k,:)=min(n(k)-1,plus(k,:));end
    ix=struct('left',1+stride*left,'right',1+stride*right,'low',facecoord(k,:)==0,'high',facecoord(k,:)==n(k),...
        'cell_low',1+fstride*low,'cell_high',1+fstride*high,'minus',1+stride*minus,'plus',1+stride*plus,...
        'at_low',cellcoord(k,:)==0,'at_high',cellcoord(k,:)==n(k)-1);
    G.index{k}=ix;G.area{k}=product(F{k});G.divergence{k}=(G.area{k}(ix.cell_high)-G.area{k}(ix.cell_low))./G.volume;
end
end

function v=product(factors)
v=1;for k=1:numel(factors),v=kron(factors{k}(:)',v);end
end

function out=coords(counts)
n=[counts,ones(1,3-numel(counts))];[I,J,K]=ndgrid(0:n(1)-1,0:n(2)-1,0:n(3)-1);
out=[I(:)';J(:)';K(:)'];out=out(1:numel(counts),:);
end
