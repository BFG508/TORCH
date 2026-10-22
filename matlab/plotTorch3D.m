function fig = plotTorch3D(result, slice_axis, slice_index, save_prefix)
% Plot a plane and its centerline profiles. MATLAB; index is one-based.
% Example: plotTorch3D(torchSolver3D('nx',12,'ny',12,'nz',12),'z');
if nargin<2, slice_axis='z'; end
axes_names={'z','y','x'}; dim=find(strcmp(axes_names,slice_axis));
assert(~isempty(dim),'Slice axis must be x, y or z');
shape=[size(result.u_final,2),size(result.u_final,3),size(result.u_final,4)];
if nargin<3 || isempty(slice_index), slice_index=floor(shape(dim)/2)+1; end
assert(isscalar(slice_index) && slice_index==fix(slice_index) && slice_index>=1 && slice_index<=shape(dim),'Invalid slice index');
other=setdiff(1:3,dim); vertical=axes_names{other(1)}; horizontal=axes_names{other(2)};
P=result.u_prim_final;
fields={reshape(P(1,:,:,:),shape),reshape(P(5,:,:,:),shape),reshape(sqrt(sum(P(2:4,:,:,:).^2,1)),shape)};
titles={'Density [kg/m^3]','Pressure [Pa]','Speed [m/s]'};
if ~isempty(result.lambda_final), fields{3}=result.lambda_final; titles{3}='Reactant fraction'; end
fig=figure('Color','white','Position',[100,100,1200,750]);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
for k=1:3
    index=repmat({':'},1,3); index{dim}=slice_index;
    field=fields{k}; plane=squeeze(field(index{:}));
    nexttile(k); imagesc(result.(horizontal),result.(vertical),plane); set(gca,'YDir','normal');
    colorbar; xlabel([horizontal ' [m]']); ylabel([vertical ' [m]']); title(titles{k});
    nexttile(k+3); plot(result.(horizontal),plane(floor(size(plane,1)/2)+1,:));
    xlabel([horizontal ' [m]']); ylabel(titles{k}); title('Slice centerline'); grid on;
end
coordinate=result.(slice_axis);
sgtitle(sprintf('3D slice: %s=%.6g m',slice_axis,coordinate(slice_index)));
if nargin>=4 && ~isempty(save_prefix)
    parent=fileparts(save_prefix); if ~isempty(parent) && ~exist(parent,'dir'), mkdir(parent); end
    exportgraphics(fig,[save_prefix '.png'],'Resolution',200);
    print(fig,[save_prefix '.svg'],'-dsvg');
end
end
