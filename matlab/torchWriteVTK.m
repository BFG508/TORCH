function torchWriteVTK(path,s)
% VTK XML RectilinearGrid, cell-centered fields and x-fastest ordering.
[~,~,extension]=fileparts(path);assert(strcmpi(extension,'.vtr'),'VTK output requires .vtr');
data=torchSessionData(s);c=s.config;dim=c.dimension;
parent=fileparts(path);if isempty(parent),parent=pwd;end;if ~exist(parent,'dir'),mkdir(parent);end
temporary=[tempname(parent) '.vtr'];fid=fopen(temporary,'w');assert(fid>=0,'Cannot create VTK file');
cleanup=onCleanup(@()cleanupFile(fid,temporary));
counts=[c.counts_xyz,zeros(1,3-dim)];extent=sprintf('0 %d 0 %d 0 %d',counts);
fprintf(fid,'<?xml version="1.0"?>\n<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian"><RectilinearGrid WholeExtent="%s"><FieldData><DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">%.17g</DataArray></FieldData><Piece Extent="%s"><CellData Scalars="rho">',extent,s.time,extent);
for key=fieldnames(data.fields)'
    fprintf(fid,'<DataArray type="Float64" Name="%s" format="ascii">',key{1});fprintf(fid,'%.17g ',data.fields.(key{1}));fprintf(fid,'</DataArray>');
end
fprintf(fid,'</CellData><Coordinates>');
for k=1:3
    names='xyz';values=0;if k<=dim,values=linspace(0,c.lengths_xyz(k),counts(k)+1);end
    fprintf(fid,'<DataArray type="Float64" Name="%s" format="ascii">',names(k));fprintf(fid,'%.17g ',values);fprintf(fid,'</DataArray>');
end
fprintf(fid,'</Coordinates></Piece></RectilinearGrid></VTKFile>');fclose(fid);fid=-1;
[ok,message]=movefile(temporary,path,'f');assert(ok,message);
end

function cleanupFile(fid,path)
try, fclose(fid); catch, end
if exist(path,'file'),delete(path);end
end
