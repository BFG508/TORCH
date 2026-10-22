function torchDesktopExport(path,s)
% Export a displayed state; restart saving uses the separate current state.
[~,~,extension]=fileparts(path);parent=fileparts(path);if ~isempty(parent)&&~exist(parent,'dir'),mkdir(parent);end
switch lower(extension)
    case '.h5',torchWriteCheckpoint(path,s);
    case {'.vtr','.vts','.vtu'},torchWriteVTK(path,s);
    case {'.xdmf','.xmf'},torchWriteXDMF(path,s);
    case '.csv'
        c=s.config;d=c.dimension;G=torchSessionGrid(c);a=cell(1,3);
        for k=1:3,if k<=d,a{k}=(G.axes{k}(1:end-1)+G.axes{k}(2:end))/2;else,a{k}=0;end;end
        [X,Y,Z]=ndgrid(a{1},a{2},a{3});columns={X(:),Y(:),Z(:)};names=torchCoordinates(c);M=[];
        for k=1:d,M=[M,columns{k}];end %#ok<AGROW>
        if isfield(s,'mesh'),M=s.mesh.centers;end
        data=torchSessionData(s);M=[M,data.fields.pressure,data.fields.rho];headers=[names,{'p','rho'}];
        for k=1:d,M=[M,data.fields.(['velocity_' names{k}])];headers{end+1}=['u_' names{k}];end %#ok<AGROW>
        if isfield(s,'mesh'),M=[M,s.mesh.widths,double(s.mesh.levels),s.mesh.volumes(:)];headers=[headers,cellfun(@(a)['width_' a],names,'UniformOutput',false),{'refinement_level','cell_volume'}];end
        if ~isempty(s.lam),M=[M,data.fields.lambda];headers{end+1}='lambda';end
        writetable(array2table(M,'VariableNames',headers),path);
    otherwise,error('Choose CSV, HDF5, VTK or XDMF output');
end
end
