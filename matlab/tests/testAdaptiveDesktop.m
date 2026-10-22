function testAdaptiveDesktop()
% Real GUI callbacks for adaptive leaf fields, changing playback and restart.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
folder=tempname;mkdir(folder);cleanupFolder=onCleanup(@()rmdir(folder,'s'));app=interface2D('Visible','off');cleanup=onCleanup(@()deleteIfValid(app));
for d=2:3
    app.reset();p=torchDesktopConfig('defaults',d);p.counts=[8,6,4];p.counts=p.counts(1:d);p.adaptive=true;p.amr.max_level=1;p.amr.regrid_every=1;p.workers=2;p.t_end=.04;app.configure(p);app.Inputs.folder.Value=folder;app.Inputs.interval.Value=1;
    expected=torchDesktopConfig('create',app.specification());expected=torchRunSession(expected);
    f=app.RunButton.ButtonPushedFcn;f(app.RunButton,[]);deadline=tic;
    while app.Running&&toc(deadline)<60,drawnow;pause(.02);end
    assert(~app.Running&&~startsWith(app.Status.Text,'Error:'),app.Status.Text);assert(isequal(app.Session.state,expected.state),'Adaptive GUI state mismatch');
    assert(strcmp(app.Inputs.amr_max_level.Enable,'off')&&strcmp(app.Inputs.workers.Enable,'on'),'AMR controls incorrectly locked');
    app.FieldDropDown.Value='Refinement level';app.redraw();assert(numel(app.LastPlane.values)<=size(app.Session.mesh.keys,1),'GUI expanded finest mesh');
    app.ModeDropDown.Value='3D plane';app.redraw();app.renderHistory(1);app.saveCurrent(fullfile(folder,'restart.h5'));app.exportDisplayed(fullfile(folder,'cells.vtu'));app.exportDisplayed(fullfile(folder,'cells.csv'));
    saved=app.Session;app.load(fullfile(folder,'restart.h5'));assert(isequal(app.Session.state,saved.state),'GUI changed leaf state on load');
    app.Inputs.workers.Value=1;app.Inputs.t_end.Value=.06;app.run();app.pause();assert(~app.Running,'Pause did not stop AMR');
end
fprintf('PASS MATLAB AMR GUI, controls, leaf rendering, restart and playback\n');
end
function deleteIfValid(app)
if isvalid(app),delete(app);end
end
