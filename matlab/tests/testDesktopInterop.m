function testDesktopInterop(folder)
% Resume Python checkpoints in the GUI; save current and displayed states.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
for dim=[2,3]
    app=torchDesktop(dim,'Visible','off');cleanup=onCleanup(@()deleteIfValid(app));
    app.Inputs.folder.Value=folder;app.Inputs.interval.Value=1;
    app.load(fullfile(folder,sprintf('python_%d.h5',dim)));
    settings=app.specification();assert(isequal(app.Session.config.counts_xyz,settings.counts),'GUI mesh was not restored');
    app.run();start=tic;while app.Running&&toc(start)<45,drawnow;pause(.02);end
    assert(~app.Running&&~startsWith(app.Status.Text,'Error:'),app.Status.Text);
    expected=torchReadCheckpoint(fullfile(folder,sprintf('expected_%d.h5',dim)));
    assert(all(abs(app.Session.state(:)-expected.state(:))<2e-8+3e-10*abs(expected.state(:))),'MATLAB desktop continuation mismatch');
    app.renderHistory(1);assert(app.Displayed.time<app.Session.time,'History frame did not change');
    app.saveCurrent(fullfile(folder,sprintf('matlab_%d.h5',dim)));
    app.exportDisplayed(fullfile(folder,sprintf('matlab_displayed_%d.h5',dim)));
    fid=fopen(fullfile(folder,sprintf('series_%d.txt',dim)),'w');fprintf(fid,'%s',app.Prefix);fclose(fid);
    clear cleanup;
end
fprintf('PASS MATLAB desktop continuation of Python 2D/3D checkpoints\n');
end

function deleteIfValid(app)
if isvalid(app),delete(app);end
end
