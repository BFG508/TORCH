function torchWriteCheckpoint(path,s,visualization)
% Schema-2 HDF5 checkpoint. Write a complete temporary file, then replace target.
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
    [points,cells]=torchMeshData(s.config);
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
