function text = torchXDMFGrid(filename,time,c,names)
% Internal XML fragment shared by snapshot and temporal XDMF writers.
types={'Polyline','Quadrilateral','Hexahedron'};nodes=[2,4,8];dim=c.dimension;
filename=escape(filename);n=prod(c.counts_xyz);points=prod(c.counts_xyz+1);
text=sprintf('<Grid GridType="Uniform"><Time Value="%.17g"/><Topology TopologyType="%s" NumberOfElements="%d" NodesPerElement="%d"><DataItem Dimensions="%d %d" NumberType="Int" Precision="8" Format="HDF">%s:/mesh/cells</DataItem></Topology><Geometry GeometryType="XYZ"><DataItem Dimensions="%d 3" NumberType="Float" Precision="8" Format="HDF">%s:/mesh/points</DataItem></Geometry>',time,types{dim},n,nodes(dim),n,nodes(dim),filename,points,filename);
for k=1:numel(names)
    name=escape(names{k});
    text=[text,sprintf('<Attribute Name="%s" AttributeType="Scalar" Center="Cell"><DataItem Dimensions="%d" NumberType="Float" Precision="8" Format="HDF">%s:/fields/%s</DataItem></Attribute>',name,n,filename,name)]; %#ok<AGROW>
end
text=[text,'</Grid>'];
end

function text=escape(text)
text=strrep(text,'&','&amp;');text=strrep(text,'<','&lt;');text=strrep(text,'>','&gt;');text=strrep(text,'"','&quot;');
end
