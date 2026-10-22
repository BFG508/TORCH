function plane = torchDesktopDraw(s,plotAxes,profileAxes,name,normal,index,mode)
% Replace artists without creating figures or retaining solution histories.
plane=torchDesktopPlane(s,name,normal,index);cla(plotAxes);cla(profileAxes);
if isappdata(plotAxes,'TORCHColorbar'),cb=getappdata(plotAxes,'TORCHColorbar');if isgraphics(cb),delete(cb);end;end
if s.config.dimension==1
    plot(plotAxes,plane.profile_x,plane.profile);xlabel(plotAxes,plane.x_label);ylabel(plotAxes,plane.field_label);grid(plotAxes,'on');
else
    if isfield(s,'mesh')
        n=numel(plane.values);faces=reshape(1:4*n,4,n)';
        if strcmp(mode,'3D plane')
            vertices=reshape(permute(plane.vertices,[2,1,3]),[],3);view(plotAxes,3);xlabel(plotAxes,'x [m]');ylabel(plotAxes,'y [m]');zlabel(plotAxes,'z [m]');
        else
            vertices=[reshape(plane.map_x',[],1),reshape(plane.map_y',[],1),zeros(4*n,1)];view(plotAxes,2);xlabel(plotAxes,plane.x_label);ylabel(plotAxes,plane.y_label);
        end
        patch(plotAxes,'Vertices',vertices,'Faces',faces,'FaceVertexCData',plane.values(:),'FaceColor','flat','EdgeColor',[.3,.3,.3],'LineWidth',.2);axis(plotAxes,'tight');
        if s.config.dimension==2&&strcmp(mode,'Map'),axis(plotAxes,'equal');end
    else
    colors=plane.values;colors(end+1,:)=colors(end,:);colors(:,end+1)=colors(:,end);
    if strcmp(mode,'3D plane')
        p=plane.vertices;surface(plotAxes,p(:,:,1),p(:,:,2),p(:,:,3),colors,'FaceColor','flat','EdgeColor','none');view(plotAxes,3);
        xlabel(plotAxes,'x [m]');ylabel(plotAxes,'y [m]');zlabel(plotAxes,'z [m]');axis(plotAxes,'tight');
    else
        surface(plotAxes,plane.map_x,plane.map_y,zeros(size(plane.map_x)),colors,'FaceColor','flat','EdgeColor','none');view(plotAxes,2);
        xlabel(plotAxes,plane.x_label);ylabel(plotAxes,plane.y_label);axis(plotAxes,'tight');if s.config.dimension==2,axis(plotAxes,'equal');pbaspect(plotAxes,[max(plane.map_x(:))-min(plane.map_x(:)),max(plane.map_y(:))-min(plane.map_y(:)),1]);end
    end
    end
    if strcmp(mode,'3D plane'),daspect(plotAxes,[1,1,1]);else,daspect(plotAxes,'auto');end
    limits=[min(plane.values(:)),max(plane.values(:))];if limits(1)==limits(2),delta=max(1,abs(limits(1)))*1e-9;limits=limits+[-delta,delta];end
    clim(plotAxes,limits);colormap(plotAxes,'parula');cb=colorbar(plotAxes);drawnow nocallbacks;cb.Label.String=plane.field_label;setappdata(plotAxes,'TORCHColorbar',cb);
    plot(profileAxes,plane.profile_x,plane.profile);xlabel(profileAxes,plane.profile_label);ylabel(profileAxes,plane.field_label);title(profileAxes,'Slice centreline');grid(profileAxes,'on');
end
title(plotAxes,sprintf('%s | t=%.6g s | step %d',plane.slice_label,s.time,s.steps));
end
