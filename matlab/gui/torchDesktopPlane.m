function out = torchDesktopPlane(s,name,normal,index)
% Cell-centred field and mapped vertices. Slice indices are one-based.
c=s.config;d=c.dimension;names=torchCoordinates(c);G=torchMetricGrid(c);centers=cell(1,d);
for k=1:d,centers{k}=(G.axes{k}(1:end-1)+G.axes{k}(2:end))/2;end
flat=reshape(s.state,d+2,[]);gamma=c.gamma;if c.reactive,gamma=(4.5-s.lam(:)')./(3.5-s.lam(:)');end
P=flat;P(2:end-1,:)=flat(2:end-1,:)./flat(1,:);P(end,:)=(gamma-1).*(flat(end,:)-.5*sum(flat(2:end-1,:).^2,1)./flat(1,:));
if strcmp(name,'Pressure'),value=P(end,:);label='Pressure [Pa]';
elseif strcmp(name,'Density'),value=P(1,:);label='Density [kg/m^3]';
elseif strcmp(name,'Speed'),value=sqrt(sum(P(2:end-1,:).^2,1));label='Speed [m/s]';
elseif startsWith(name,'Velocity '),k=find(strcmp(names,name(10:end)));assert(~isempty(k),'Unavailable velocity');value=P(k+1,:);label=[name ' [m/s]'];
elseif strcmp(name,'Reactant fraction') && ~isempty(s.lam),value=s.lam(:)';label='Unburned fraction';
else,error('Unavailable field');end
if d==1
    out=struct('values',value,'vertices',[],'map_x',G.axes{1},'map_y',zeros(size(G.axes{1})),...
        'profile_x',centers{1},'profile',value,'x_label',coordinateLabel(names{1}),'y_label','',...
        'field_label',label,'slice_label','Profile');return;
end
shape=c.counts_xyz(end:-1:1);value=reshape(value,shape);remaining=1:d;selected='2D field';
if d==3
    assert(any(normal==[1,2,3]),'Invalid slice axis');count=c.counts_xyz(normal);
    if nargin<4 || isempty(index),index=floor(count/2)+1;end
    assert(isscalar(index)&&index==fix(index)&&index>=1&&index<=count,'Invalid slice index');
    pick={':',':',':'};pick{4-normal}=index;value=squeeze(value(pick{:}));remaining(normal)=[];
    selected=sprintf('%s = %.6g',names{normal},centers{normal}(index));
end
horizontal=remaining(1);vertical=remaining(2);
[xx,yy]=meshgrid(G.axes{horizontal},G.axes{vertical});logical=zeros(numel(xx),3);
logical(:,horizontal)=xx(:);logical(:,vertical)=yy(:);if d==3,logical(:,normal)=centers{normal}(index);end
points=torchMapPoints(c,logical);vertices=reshape(points,[size(xx),3]);
mapx=xx;mapy=yy;xlabel=coordinateLabel(names{horizontal});ylabel=coordinateLabel(names{vertical});
if d==2 && isfield(c,'geometry') && ~strcmp(c.geometry,'cartesian')
    pair=[1,2];if any(strcmp(c.geometry,{'cylindrical','spherical'})),pair=[1,3];end
    mapx=vertices(:,:,pair(1));mapy=vertices(:,:,pair(2));xyz={'x [m]','y [m]','z [m]'};xlabel=xyz{pair(1)};ylabel=xyz{pair(2)};
end
out=struct('values',value,'vertices',vertices,'map_x',mapx,'map_y',mapy,...
    'profile_x',centers{horizontal},'profile',value(floor(size(value,1)/2)+1,:),...
    'x_label',xlabel,'y_label',ylabel,'field_label',label,'slice_label',selected,...
    'profile_label',coordinateLabel(names{horizontal}));
end

function value=coordinateLabel(name)
unit='m';if any(strcmp(name,{'theta','phi'})),unit='rad';end;value=[name ' [' unit ']'];
end
