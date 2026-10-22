function torchWriteXDMF(path,s)
% Standalone XDMF/HDF5 cell-centered snapshot, also a restart checkpoint.
[parent,name,extension]=fileparts(path);assert(any(strcmpi(extension,{'.xdmf','.xmf'})),'XDMF output requires .xdmf or .xmf');hdf=fullfile(parent,[name '.h5']);
torchWriteCheckpoint(hdf,s,true);data=torchSessionData(s);
grid=torchXDMFGrid([name '.h5'],s.time,s.config,fieldnames(data.fields));
grid=strrep(grid,sprintf('<Time Value="%.17g"/>',s.time),'');
text=['<?xml version="1.0"?><Xdmf Version="3.0"><Domain>',grid,'</Domain></Xdmf>'];
torchAtomicText(path,text);
end
