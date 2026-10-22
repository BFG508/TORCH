function torchWriteCheckpoint(path,s,visualization)
% Schema-1 HDF5 checkpoint. Write a complete temporary file, then replace target.
if nargin<3,visualization=false;end
data=torchSessionData(s);parent=fileparts(path);if isempty(parent),parent=pwd;end
if ~exist(parent,'dir'),mkdir(parent);end
temporary=[tempname(parent) '.h5']; cleanup=onCleanup(@()removeTemporary(temporary));
bytes=unicode2native(jsonencode(data.metadata),'UTF-8');
h5create(temporary,'/metadata',numel(bytes),'Datatype','uint8');h5write(temporary,'/metadata',bytes(:));
for name=data.names
    values=data.fields.(name{1});h5create(temporary,['/state/' name{1}],numel(values),'Datatype','double');h5write(temporary,['/state/' name{1}],values);
end
for name={'lambda','beta'}
    if isfield(data.fields,name{1})
        values=data.fields.(name{1});h5create(temporary,['/state/' name{1}],numel(values),'Datatype','double');h5write(temporary,['/state/' name{1}],values);
    end
end
if visualization
    [points,cells]=meshData(s.config);
    h5create(temporary,'/mesh/points',fliplr(size(points)),'Datatype','double');h5write(temporary,'/mesh/points',points');
    h5create(temporary,'/mesh/cells',fliplr(size(cells)),'Datatype','int64');h5write(temporary,'/mesh/cells',int64(cells'));
    names=fieldnames(data.fields);
    for k=1:numel(names)
        values=data.fields.(names{k});h5create(temporary,['/fields/' names{k}],numel(values),'Datatype','double');h5write(temporary,['/fields/' names{k}],values);
    end
end
[ok,message]=movefile(temporary,path,'f');assert(ok,message);
end

function removeTemporary(path)
if exist(path,'file'),delete(path);end
end

function [points,cells]=meshData(c)
n=[c.counts_xyz,ones(1,3-c.dimension)];lengths=[c.lengths_xyz,zeros(1,3-c.dimension)];axes=cell(1,3);
for k=1:3,if k<=c.dimension,axes{k}=linspace(0,lengths(k),n(k)+1);else,axes{k}=0;end;end
[Z,Y,X]=ndgrid(axes{3},axes{2},axes{1});
X=permute(X,[3,2,1]);Y=permute(Y,[3,2,1]);Z=permute(Z,[3,2,1]);points=[X(:),Y(:),Z(:)];
[K,J,I]=ndgrid(0:n(3)-1,0:n(2)-1,0:n(1)-1);
nyNodes=1;if c.dimension>=2,nyNodes=n(2)+1;end
base=I+(n(1)+1)*(J+nyNodes*K);base=permute(base,[3,2,1]);
if c.dimension==1,offsets=[0,1];
elseif c.dimension==2,offsets=[0,1,n(1)+2,n(1)+1];
else,plane=(n(1)+1)*(n(2)+1);offsets=[0,1,n(1)+2,n(1)+1,plane,plane+1,plane+n(1)+2,plane+n(1)+1];end
cells=base(:)+offsets;
end
