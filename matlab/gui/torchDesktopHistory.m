function data = torchDesktopHistory(prefix)
% Validate a disk-backed collection before playback or continuation.
data=jsondecode(fileread([prefix '.series.json']));formats=data.formats;if ischar(formats),formats={formats};end
assert(data.schema_version==1 && any(strcmp(formats,'xdmf')),'Playback requires HDF5/XDMF frames');
frames=data.frames;assert(isstruct(frames)&&~isempty(frames),'No recorded frames');parent=fileparts(prefix);time=-1;step=-1;
for k=1:numel(frames)
    f=frames(k);[folder,name,extension]=fileparts(f.stem);
    assert(isempty(folder)&&strcmp([name extension],f.stem),'Invalid frame path');
    assert(exist(fullfile(parent,[f.stem '.h5']),'file')==2,'Missing recorded frame');
    assert(isfinite(f.time)&&f.time>time&&f.step>step,'Unordered frame collection');time=f.time;step=f.step;
end
end
