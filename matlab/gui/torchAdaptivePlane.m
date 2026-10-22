function out=torchAdaptivePlane(s,name,normal,index)
% Render actual leaf polygons, never a dense maximum-level grid.
c=s.config;d=c.dimension;M=s.mesh;names=torchCoordinates(c);[P,~]=torchAdaptivePrimitive(s.state,s.lam,c);
if strcmp(name,'Pressure'),value=P(end,:);label='Pressure [Pa]';
elseif strcmp(name,'Density'),value=P(1,:);label='Density [kg/m^3]';
elseif strcmp(name,'Speed'),value=sqrt(sum(P(2:end-1,:).^2,1));label='Speed [m/s]';
elseif strcmp(name,'Refinement level'),value=M.levels';label='Refinement level';
elseif strcmp(name,'Cell volume'),value=M.volumes;label='Metric cell volume';
elseif startsWith(name,'Velocity '),k=find(strcmp(names,name(10:end)));assert(~isempty(k),'Unavailable velocity');value=P(k+1,:);label=[name ' [m/s]'];
elseif strcmp(name,'Reactant fraction')&&~isempty(s.lam),value=s.lam;label='Unburned fraction';
else,error('Unavailable adaptive field');end
if d==1
    [x,order]=sort(M.centers(:,1));out=struct('values',value(order),'vertices',[],'map_x',x,'map_y',zeros(size(x)),'profile_x',x,'profile',value(order),'x_label',coordinateLabel(names{1}),'y_label','','field_label',label,'slice_label','Adaptive profile');return;
end
remaining=1:d;mask=true(size(M.keys,1),1);selected='Adaptive 2D field';position=0;
if d==3
    assert(isscalar(normal)&&any(normal==1:3),'Invalid slice axis');count=c.counts_xyz(normal)*2^c.amr.max_level;
    if nargin<4||isempty(index),index=floor(count/2)+1;end
    assert(isscalar(index)&&index==fix(index)&&index>=1&&index<=count,'Invalid slice index');
    position=c.origins_xyz(normal)+(index-.5)*c.lengths_xyz(normal)/count;
    mask=M.low(:,normal)<=position&position<M.high(:,normal);remaining(normal)=[];selected=sprintf('Adaptive slice: %s = %.6g',names{normal},position);
end
h=remaining(1);v=remaining(2);lo=M.low(mask,:);hi=M.high(mask,:);n=size(lo,1);logical=zeros(n*4,3);
logical(:,h)=reshape([lo(:,h),hi(:,h),hi(:,h),lo(:,h)]',[],1);logical(:,v)=reshape([lo(:,v),lo(:,v),hi(:,v),hi(:,v)]',[],1);
if d==3,logical(:,normal)=position;end
points=torchMapPoints(c,logical);vertices=permute(reshape(points,4,n,3),[2,1,3]);mx=reshape(logical(:,h),4,n)';my=reshape(logical(:,v),4,n)';xl=coordinateLabel(names{h});yl=coordinateLabel(names{v});
if d==2&&~strcmp(c.geometry,'cartesian')
    pair=[1,2];if any(strcmp(c.geometry,{'cylindrical','spherical'})),pair=[1,3];end
    mx=vertices(:,:,pair(1));my=vertices(:,:,pair(2));xyz='xyz';xl=[xyz(pair(1)) ' [m]'];yl=[xyz(pair(2)) ' [m]'];
end
middle=c.origins_xyz(v)+.5*c.lengths_xyz(v);ids=find(mask&M.low(:,v)<=middle&middle<M.high(:,v));[~,order]=sort(M.centers(ids,h));ids=ids(order);
out=struct('values',value(mask),'vertices',vertices,'map_x',mx,'map_y',my,'profile_x',M.centers(ids,h),'profile',value(ids),'x_label',xl,'y_label',yl,'field_label',label,'slice_label',selected,'profile_label',coordinateLabel(names{h}));
end
function value=coordinateLabel(name)
unit='m';if any(strcmp(name,{'theta','phi'})),unit='rad';end;value=[name ' [' unit ']'];
end
