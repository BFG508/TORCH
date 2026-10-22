function testDesktop(folder)
% Exercise real 2D/3D GUI callbacks, cooperative timers, restart and playback.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
if nargin<1,folder=tempname;mkdir(folder);cleanupFolder=onCleanup(@()rmdir(folder,'s'));end
app=interface2D('Visible','off');cleanup=onCleanup(@()deleteIfValid(app));
app.Inputs.folder.Value=folder;app.Inputs.nx.Value=12;app.Inputs.ny.Value=8;app.Inputs.t_end.Value=.04;app.Inputs.interval.Value=1;
expected=torchDesktopConfig('create',app.specification());expected=torchRunSession(expected);
f=app.RunButton.ButtonPushedFcn;f(app.RunButton,[]);waitForIdle(app,30);
assert(isequal(app.Session.state,expected.state),'Desktop result differs from session API');
assert(numel(app.Frames)>=2 && strcmp(app.Inputs.nx.Enable,'off'),'Missing recorded history/config lock');
app.renderHistory(1);assert(app.Displayed.time==0,'History did not select initial frame');
app.saveCurrent(fullfile(folder,'desktop_current.h5'));saved=torchReadCheckpoint(fullfile(folder,'desktop_current.h5'));assert(saved.time==expected.time,'Saved viewed frame instead of latest state');
app.exportDisplayed(fullfile(folder,'desktop_displayed.csv'));app.exportFigure(fullfile(folder,'desktop_displayed.png'));
T=readtable(fullfile(folder,'desktop_displayed.csv'));plane=torchDesktopPlane(app.Displayed,'Pressure',2,1);assert(max(abs(sort(T.p)-sort(plane.values(:))))<1e-12,'CSV does not match displayed frame');
app.Inputs.t_end.Value=.08;saved.config.t_end=.08;saved=torchRunSession(saved);app.run();waitForIdle(app,30);assert(isequal(app.Session.state,saved.state),'Resume did not continue latest state');
app.play();drawnow;pause(.4);app.stopPlayback();assert(isempty(app.PlayTimer),'Playback timer was not stopped');
app.reset();assert(isempty(app.Session)&&strcmp(app.Inputs.nx.Enable,'on'),'New did not release configuration');
app.Inputs.nx.Value=3.5;app.run();assert(startsWith(app.Status.Text,'Error:')&&~app.Running&&isempty(app.Session),'Invalid input was accepted');
app.Inputs.nx.Value=12;app.Inputs.t_end.Value=1;app.run();deadline=tic;
while app.Session.steps<1&&app.Running&&toc(deadline)<20,drawnow;pause(.01);end
f=app.PauseButton.ButtonPushedFcn;f(app.PauseButton,[]);assert(~app.Running&&app.Session.time<1,'Pause did not stop between steps');assert(isempty(app.StepTimer),'Step timer leaked');
series=app.Prefix;app.saveCurrent(fullfile(folder,'desktop_paused.h5'));paused=app.Session;app.reset();app.load(fullfile(folder,'desktop_paused.h5'));assert(isequal(app.Session.state,paused.state),'Checkpoint GUI load changed state');
original=app.Session;bad=fullfile(folder,'desktop_bad.h5');fid=fopen(bad,'w');fprintf(fid,'not HDF5');fclose(fid);
try,app.load(bad);catch,end;assert(isequal(app.Session.state,original.state),'Bad load destroyed the current state');
app.load([series '.series.json'],true);assert(numel(app.Frames)>=2,'Series GUI load failed');
app.reset();p=torchDesktopConfig('defaults',3,'spherical');p.counts=[8,6,4];p.t_end=1e-6;p.pressure=1e6;p.reactive=true;app.configure(p);app.run();waitForIdle(app,30);
assert(any(strcmp(app.FieldDropDown.Items,'Reactant fraction')),'Missing reactive field');
app.AxisDropDown.Value='r';app.updateSlice(true);app.SliceSlider.Value=6;app.FieldDropDown.Value='Density';app.redraw();
P=reshape(app.Displayed.state(1,:,:,:),4,6,8);expected=squeeze(P(:,:,6));assert(max(abs(app.LastPlane.values(:)-expected(:)))<1e-12,'GUI slice differs from state');
app.ModeDropDown.Value='3D plane';app.redraw();radius=sqrt(sum(app.LastPlane.vertices.^2,3));assert(max(abs(radius(:)-(.5+5.5*.5/8)))<1e-12,'3D surface is not in physical coordinates');
app.exportDisplayed(fullfile(folder,'desktop_3d.xdmf'));app.exportDisplayed(fullfile(folder,'desktop_3d.vts'));
app.exportFigure(fullfile(folder,'desktop_3d.svg'));
app.reset();app.configure(torchDesktopConfig('defaults',2));app.Inputs.folder.Value=folder;app.Inputs.t_end.Value=1;app.run();delete(app);assert(~isvalid(app),'Closing a running GUI did not clean up');
clear cleanup;
fprintf('PASS MATLAB 2D/3D GUI, pause/resume, playback, export and error handling\n');
end

function waitForIdle(app,seconds)
start=tic;while app.Running&&toc(start)<seconds,drawnow;pause(.02);end
assert(~app.Running,'Desktop did not return to idle');assert(~startsWith(app.Status.Text,'Error:'),app.Status.Text);
end

function deleteIfValid(app)
if isvalid(app),delete(app);end
end
