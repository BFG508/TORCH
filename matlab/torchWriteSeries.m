function torchWriteSeries(prefix,s,formats)
% Append one frame on disk. A manifest tracks metadata, never solution arrays.
if nargin<3,formats={'vtk','xdmf'};end
assert(iscell(formats) && ~isempty(formats) && all(ismember(formats,{'vtk','xdmf'})),'Unsupported output formats');
manifest=[prefix '.series.json'];
if exist(manifest,'file')
    data=jsondecode(fileread(manifest));
    assert(strcmp(data.run_id,s.run_id) && isequal(data.formats(:),formats(:)),'Output prefix belongs to another run or formats');
    keys=fieldnames(s.config);
    for k=1:numel(keys)
        key=keys{k};if any(strcmp(key,{'t_end','max_steps'})),continue;end
        a=data.config.(key);b=s.config.(key);
        if isnumeric(a) || iscell(a),a=a(:);b=b(:);end
        assert(isequal(a,b),'Output series configuration mismatch');
    end
    frames=data.frames;
    if ~isempty(frames)
        assert(frames(end).time<=s.time,'Series is ahead of checkpoint; choose a new prefix');
        if frames(end).time==s.time,assert(frames(end).step==s.steps,'Duplicate time at a different step');return;end
    end
else
    data=struct('schema_version',1,'run_id',s.run_id,'formats',{formats},'config',s.config,'frames',[]);frames=[];
end
[parent,name]=fileparts(prefix);stem=sprintf('%s_%08d',name,s.steps);path=fullfile(parent,stem);
if any(strcmp(formats,'vtk')),torchWriteVTK([path '.vtr'],s);end
if any(strcmp(formats,'xdmf')),torchWriteCheckpoint([path '.h5'],s,true);end
snapshot=torchSessionData(s);frame=struct('time',s.time,'step',s.steps,'stem',stem,'fields',{fieldnames(snapshot.fields)});
if isempty(frames),frames=frame;else,frames(end+1)=frame;end
data.frames=frames;
if any(strcmp(formats,'vtk'))
    text='<?xml version="1.0"?><VTKFile type="Collection" version="1.0" byte_order="LittleEndian"><Collection>';
    for k=1:numel(frames)
        filename=strrep(strrep([frames(k).stem '.vtr'],'&','&amp;'),'"','&quot;');
        text=[text,sprintf('<DataSet timestep="%.17g" group="" part="0" file="%s"/>',frames(k).time,filename)]; %#ok<AGROW>
    end
    torchAtomicText([prefix '.pvd'],[text,'</Collection></VTKFile>']);
end
if any(strcmp(formats,'xdmf'))
    text='<?xml version="1.0"?><Xdmf Version="3.0"><Domain><Grid Name="TORCH" GridType="Collection" CollectionType="Temporal">';
    for k=1:numel(frames)
        text=[text,torchXDMFGrid([frames(k).stem '.h5'],frames(k).time,s.config,frames(k).fields)]; %#ok<AGROW>
    end
    torchAtomicText([prefix '.xdmf'],[text,'</Grid></Domain></Xdmf>']);
end
% jsonencode scalar structs as an array too: frames is always a JSON list.
encoded=jsonencode(data);
if numel(frames)==1
    encoded=strrep(encoded,['"frames":' jsonencode(frames)],['"frames":[' jsonencode(frames) ']']);
end
torchAtomicText(manifest,encoded);
end
