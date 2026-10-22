function torchWriteAdaptiveVTK(path,s)
% Portable unstructured line/quad/hex output, one cell per AMR leaf.
[~,~,ext]=fileparts(path);assert(strcmpi(ext,'.vtu'),'Adaptive VTK requires .vtu');
data=torchAdaptiveData(s);[points,cells]=torchAdaptiveMeshData(s);n=size(cells,1);nodes=size(cells,2);types=[3,9,12];
text=sprintf('<?xml version="1.0"?><VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian"><UnstructuredGrid><FieldData><DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">%.17g</DataArray></FieldData><Piece NumberOfPoints="%d" NumberOfCells="%d"><CellData Scalars="rho">',s.time,size(points,1),n);
for key=fieldnames(data.fields)',text=[text,array(key{1},'Float64',data.fields.(key{1}),'%.17g ')];end %#ok<AGROW>
text=[text,'</CellData><Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">',sprintf('%.17g ',points'),'</DataArray></Points><Cells>',array('connectivity','Int64',cells','%d '),array('offsets','Int64',(1:n)*nodes,'%d '),array('types','UInt8',repmat(types(s.config.dimension),1,n),'%d '),'</Cells></Piece></UnstructuredGrid></VTKFile>'];
torchAtomicText(path,text);
end
function text=array(name,type,values,format)
text=['<DataArray type="',type,'" Name="',name,'" format="ascii">',sprintf(format,values),'</DataArray>'];
end
