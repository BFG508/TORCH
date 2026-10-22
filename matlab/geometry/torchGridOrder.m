function out = torchGridOrder(values,c,action)
% Convert [components,reversed spatial axes] to/from [components,x-fastest].
d=c.dimension;nv=size(values,1);
if d==1,out=values;return;end
if strcmp(action,'flatten')
    out=reshape(permute(reshape(values,[nv,c.counts_xyz(end:-1:1)]),[1,d+1:-1:2]),nv,[]);
else
    out=permute(reshape(values,[nv,c.counts_xyz]),[1,d+1:-1:2]);
end
end
