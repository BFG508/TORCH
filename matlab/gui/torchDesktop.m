classdef torchDesktop < handle
    % Session desktop: cooperative step timer, bounded RAM and disk playback.
    properties
        Figure
        Labels = struct
        Inputs = struct
        ConfigControls = {}
        Session = []
        Displayed = []
        Prefix = ''
        Frames = []
        FaceEdits = struct
        PlotAxes
        ProfileAxes
        LastPlane = []
        StepTimer = []
        PlayTimer = []
        Running = false
        Busy = false
        Adjusting = false
        Closing = false
        RunButton
        PauseButton
        NewButton
        FileButtons = {}
        PlayButton
        LatestButton
        FrameSlider
        SliceSlider
        FieldDropDown
        AxisDropDown
        ModeDropDown
        FaceDropDown
        FaceTypeDropDown
        BackflowCheck
        ApplyFaceButton
        Status
        Audit
        ViewTime
        LastRecorded = -1
    end
    methods
        function app=torchDesktop(dimension,varargin)
            if nargin<1,dimension=2;end
            p=inputParser;addParameter(p,'Visible','on');parse(p,varargin{:});
            app.buildUI(p.Results.Visible);
            app.configure(torchDesktopConfig('defaults',dimension));
            app.Figure.CloseRequestFcn=@(~,~)delete(app);
        end

        function configure(app,spec)
            assert(~app.Running,'Pause before changing setup');app.Adjusting=true;
            if spec.dimension==1,app.Inputs.dimension.Items={'1D','2D','3D'};else,app.Inputs.dimension.Items={'2D','3D'};end
            app.Inputs.dimension.Value=sprintf('%dD',spec.dimension);
            keys={'geometry','t_end','cfl','gamma','limiter','density','pressure','rotation','fraction'};
            for k=1:numel(keys),app.Inputs.(keys{k}).Value=spec.(keys{k});end
            app.Inputs.preset.Items=torchDesktopConfig('presets',max(2,spec.dimension),spec.geometry);app.Inputs.preset.Value=spec.preset;
            app.Inputs.reactive.Value=spec.reactive;
            for k=1:3
                axis='xyz';vel='uvw';count=2;extent=1;origin=0;speed=0;
                if k<=spec.dimension,count=spec.counts(k);extent=spec.lengths(k);origin=spec.origins(k);end
                if k<=numel(spec.velocities),speed=spec.velocities(k);end
                app.Inputs.(['n' axis(k)]).Value=count;app.Inputs.(['l' axis(k)]).Value=extent;app.Inputs.(['o' axis(k)]).Value=origin;app.Inputs.(vel(k)).Value=speed;
            end
            app.FaceEdits=spec.face_edits;keys=torchDesktopConfig('keys',spec.dimension);names=torchCoordinates(struct('dimension',spec.dimension,'geometry',spec.geometry));labels={};
            app.FaceDropDown.ItemsData={};
            for k=1:spec.dimension
                labels=[labels,{[names{k} ' minimum'],[names{k} ' maximum']}];unit='m';if any(strcmp(names{k},{'theta','phi'})),unit='rad';end
                axis='xyz';vel='uvw';app.Labels.(['n' axis(k)]).Text=['Cells ' names{k}];app.Labels.(['l' axis(k)]).Text=['Extent ' names{k} ' [' unit ']'];app.Labels.(['o' axis(k)]).Text=['Origin ' names{k} ' [' unit ']'];app.Labels.(vel(k)).Text=['Velocity ' names{k} ' [m/s]'];app.Labels.(['face_' vel(k)]).Text=['Velocity ' names{k} ' [m/s]'];
            end
            app.FaceDropDown.Items=labels;app.FaceDropDown.ItemsData=keys;app.FaceDropDown.Value='x_min';
            app.Adjusting=false;app.updateControls();app.showFace();
        end

        function spec=specification(app)
            d=str2double(app.Inputs.dimension.Value(1));spec=torchDesktopConfig('defaults',max(2,d),app.Inputs.geometry.Value);spec.dimension=d;
            keys={'geometry','preset','t_end','cfl','gamma','limiter','density','pressure','rotation','fraction','reactive'};
            for k=1:numel(keys),spec.(keys{k})=app.Inputs.(keys{k}).Value;end
            spec.counts=zeros(1,d);spec.lengths=zeros(1,d);spec.origins=zeros(1,d);spec.velocities=zeros(1,d);
            for k=1:d
                axis='xyz';vel='uvw';spec.counts(k)=app.Inputs.(['n' axis(k)]).Value;spec.lengths(k)=app.Inputs.(['l' axis(k)]).Value;
                spec.origins(k)=app.Inputs.(['o' axis(k)]).Value;spec.velocities(k)=app.Inputs.(vel(k)).Value;
            end
            spec.face_edits=app.FaceEdits;
        end

        function run(app)
            if app.Running,return;end
            app.stopPlayback();
            try
                interval=app.Inputs.interval.Value;assert(interval>=1&&interval==fix(interval),'Recording interval must be a positive integer');
                if isempty(app.Session)
                    candidate=torchDesktopConfig('create',app.specification());prefix=app.freshPrefix();
                else
                    candidate=app.Session;target=app.Inputs.t_end.Value;
                    assert(isfinite(target)&&target>=candidate.time,'End time cannot precede the current state');
                    assert(target>candidate.time,'Set a later end time, or choose New');candidate.config.t_end=target;
                    prefix=app.Prefix;if isempty(prefix),prefix=app.freshPrefix();end
                end
                app.Session=candidate;app.Prefix=prefix;app.record();app.LastRecorded=app.Session.steps;
                app.Running=true;app.updateControls();app.Status.Text='Running; Pause stops between complete steps.';
                app.StepTimer=timer('ExecutionMode','fixedSpacing','Period',.05,'BusyMode','drop','TimerFcn',@(~,~)app.tick(),...
                    'ErrorFcn',@(~,event)app.timerError(event));start(app.StepTimer);
            catch ME,app.fail(ME);app.updateControls();end
        end

        function tick(app)
            if app.Busy||~app.Running||app.Closing,return;end
            app.Busy=true;cleanup=onCleanup(@()app.releaseBusy());
            try
                app.Session=torchRunSession(app.Session,'step_limit',1);
                app.Status.Text=sprintf('Running: t=%.6g s, step %d',app.Session.time,app.Session.steps);
                if mod(app.Session.steps,app.Inputs.interval.Value)==0
                    app.record();app.LastRecorded=app.Session.steps;app.showSession(app.Session);
                end
                if app.Session.time==app.Session.config.t_end,app.finish('Completed');end
            catch ME
                app.stopStepTimer();app.Running=false;app.fail(ME);app.updateControls();
            end
        end

        function pause(app)
            if ~app.Running,return;end
            app.finish('Paused; ready to resume');
        end

        function finish(app,message)
            app.stopStepTimer();app.Running=false;
            try
                app.record();app.refreshHistory();app.showSession(app.Session);
                app.Status.Text=[message '. Results: ' fileparts(app.Prefix)];
            catch ME,app.fail(ME);end
            app.updateControls();
        end

        function reset(app)
            if app.Running,return;end
            app.stopPlayback();app.Session=[];app.Displayed=[];app.Prefix='';app.Frames=[];app.LastRecorded=-1;
            cla(app.PlotAxes);cla(app.ProfileAxes);colorbar(app.PlotAxes,'off');app.LastPlane=[];app.FrameSlider.Limits=[0,1];app.FrameSlider.Value=0;
            if strcmp(app.Inputs.dimension.Value,'1D'),app.configure(torchDesktopConfig('defaults',2));end
            app.Audit.Text='No results yet';app.ViewTime.Text='No recorded frames';app.Status.Text='Configure a new simulation, then Run.';app.updateControls();
        end

        function load(app,path,series)
            if nargin<3,series=false;end
            assert(~app.Running,'Pause before loading another simulation');app.stopPlayback();
            if series
                suffix='.series.json';assert(endsWith(path,suffix),'Choose a .series.json collection');prefix=path(1:end-numel(suffix));data=torchDesktopHistory(prefix);
                candidate=torchReadCheckpoint(fullfile(fileparts(prefix),[data.frames(end).stem '.h5']));
                assert(strcmp(candidate.run_id,data.run_id),'Series identity does not match final frame');
            else,candidate=torchReadCheckpoint(path);prefix=app.freshPrefix();end
            app.reset();app.configure(torchDesktopConfig('fromSession',candidate));app.Session=candidate;app.Prefix=prefix;
            if series,app.refreshHistory();end
            app.showSession(candidate);app.updateControls();app.Status.Text='Loaded. Resume uses the saved mesh and physics.';
        end

        function showSession(app,s)
            center=isempty(app.Displayed)||app.Displayed.config.dimension~=s.config.dimension;
            app.Displayed=s;names=torchCoordinates(s.config);choices=[{'Pressure','Density','Speed'},cellfun(@(a)['Velocity ' a],names,'UniformOutput',false)];
            if ~isempty(s.lam),choices{end+1}='Reactant fraction';end
            selected=app.FieldDropDown.Value;app.FieldDropDown.Items=choices;if any(strcmp(selected,choices)),app.FieldDropDown.Value=selected;else,app.FieldDropDown.Value='Pressure';end
            app.AxisDropDown.Enable=onoff(s.config.dimension==3);selected=app.AxisDropDown.Value;app.AxisDropDown.Items=names;if any(strcmp(selected,names)),app.AxisDropDown.Value=selected;else,app.AxisDropDown.Value=names{end};end
            app.updateSlice(center);current=torchSessionIntegral(s);residual=current-s.initial_integral+s.boundary_integral-s.source_integral;
            app.Audit.Text=sprintf('Displayed mass: %.6g | relative mass balance: %.3e | energy balance: %.3e',current(1),residual(1)/max(abs(s.initial_integral(1)),1e-15),residual(end));
            app.ViewTime.Text=sprintf('Displayed: t=%.6g s, step %d',s.time,s.steps);
        end

        function updateSlice(app,center)
            if nargin<2,center=false;end
            if isempty(app.Displayed),return;end
            c=app.Displayed.config;names=torchCoordinates(c);normal=find(strcmp(names,app.AxisDropDown.Value));n=c.counts_xyz(normal);
            app.Adjusting=true;app.SliceSlider.Limits=[1,max(2,n)];index=min(n,max(1,round(app.SliceSlider.Value)));if center,index=floor(n/2)+1;end;app.SliceSlider.Value=index;app.SliceSlider.MajorTicks=unique(round(linspace(1,n,min(5,n))));
            app.SliceSlider.Enable='off';if c.dimension==3,app.SliceSlider.Enable='on';end;app.Adjusting=false;app.redraw();
        end

        function redraw(app)
            if isempty(app.Displayed)||app.Adjusting,return;end
            try
                names=torchCoordinates(app.Displayed.config);normal=find(strcmp(names,app.AxisDropDown.Value));
                app.LastPlane=torchDesktopDraw(app.Displayed,app.PlotAxes,app.ProfileAxes,app.FieldDropDown.Value,normal,round(app.SliceSlider.Value),app.ModeDropDown.Value);
            catch ME,app.fail(ME);end
        end

        function refreshHistory(app)
            data=torchDesktopHistory(app.Prefix);app.Frames=data.frames;app.Adjusting=true;
            app.FrameSlider.Limits=[1,max(2,numel(app.Frames))];app.FrameSlider.Value=numel(app.Frames);app.FrameSlider.MajorTicks=[];app.Adjusting=false;
        end

        function renderHistory(app,index)
            if app.Running||isempty(app.Frames),return;end
            index=min(numel(app.Frames),max(1,round(index)));
            try
                s=torchReadCheckpoint(fullfile(fileparts(app.Prefix),[app.Frames(index).stem '.h5']));
                app.Adjusting=true;app.FrameSlider.Value=index;app.Adjusting=false;app.showSession(s);
            catch ME,app.stopPlayback();app.fail(ME);end
        end

        function play(app)
            if ~isempty(app.PlayTimer)&&isvalid(app.PlayTimer),app.stopPlayback();return;end
            if app.Running||isempty(app.Frames),return;end
            app.PlayButton.Text='Pause playback';app.PlayTimer=timer('ExecutionMode','fixedSpacing','Period',.25,'BusyMode','drop','TimerFcn',@(~,~)app.nextFrame());start(app.PlayTimer);
        end

        function nextFrame(app)
            if app.Running||isempty(app.Frames),app.stopPlayback();return;end
            index=round(app.FrameSlider.Value)+1;if index>numel(app.Frames),index=1;end;app.renderHistory(index);
        end

        function stopPlayback(app)
            if ~isempty(app.PlayTimer)&&isvalid(app.PlayTimer),stop(app.PlayTimer);delete(app.PlayTimer);end
            app.PlayTimer=[];if ~isempty(app.PlayButton)&&isvalid(app.PlayButton),app.PlayButton.Text='Play';end
        end

        function showFace(app)
            try
                p=app.specification();keys=torchDesktopConfig('keys',p.dimension);position=find(strcmp(keys,app.FaceDropDown.Value));faces=torchDesktopConfig('faces',p);f=faces{position};
                app.FaceTypeDropDown.Value=f.type;d=ceil(position/2);primitive=[1,zeros(1,p.dimension),101325];primitive(d+1)=10;if mod(position,2)==0,primitive(d+1)=-10;end
                if isfield(f,'primitive'),primitive=f.primitive;elseif isfield(f,'backflow'),primitive=f.backflow;end
                app.Inputs.face_density.Value=primitive(1);app.Inputs.face_pressure.Value=primitive(end);
                for k=1:p.dimension,names='uvw';app.Inputs.(['face_' names(k)]).Value=primitive(k+1);end
                if strcmp(f.type,'pressure_outlet'),app.Inputs.face_pressure.Value=f.pressure;end
                parameters={'face_p0','total_pressure',110000;'face_t0','total_temperature',300;'face_r','gas_constant',287;'face_lambda','lambda',1};
                for k=1:size(parameters,1),value=parameters{k,3};if isfield(f,parameters{k,2}),value=f.(parameters{k,2});end;app.Inputs.(parameters{k,1}).Value=value;end
                app.BackflowCheck.Value=isfield(f,'backflow');app.updateFaceControls();
            catch ME,app.Status.Text=['Complete mesh inputs before editing faces: ' ME.message];end
        end

        function applyFace(app)
            try
                p=app.specification();key=app.FaceDropDown.Value;kind=app.FaceTypeDropDown.Value;f=struct('type',kind);
                if strcmp(kind,'inflow') || (strcmp(kind,'pressure_outlet')&&app.BackflowCheck.Value)
                    primitive=[app.Inputs.face_density.Value,zeros(1,p.dimension),app.Inputs.face_pressure.Value];
                    for k=1:p.dimension,names='uvw';primitive(k+1)=app.Inputs.(['face_' names(k)]).Value;end
                    if strcmp(kind,'inflow'),f.primitive=primitive;else,f.backflow=primitive;end
                end
                if strcmp(kind,'pressure_outlet'),f.pressure=app.Inputs.face_pressure.Value;end
                if strcmp(kind,'stagnation_inlet'),f.total_pressure=app.Inputs.face_p0.Value;f.total_temperature=app.Inputs.face_t0.Value;f.gas_constant=app.Inputs.face_r.Value;end
                if any(strcmp(kind,{'inflow','pressure_outlet','stagnation_inlet'})),f.lambda=app.Inputs.face_lambda.Value;end
                edits=app.FaceEdits;edits.(key)=f;opposite=[key(1:2) 'min'];if endsWith(key,'min'),opposite=[key(1:2) 'max'];end
                keys=torchDesktopConfig('keys',p.dimension);existing=torchDesktopConfig('faces',p);oppositeIndex=find(strcmp(keys,opposite));
                if strcmp(kind,'periodic'),edits.(opposite)=struct('type','periodic');
                elseif strcmp(existing{oppositeIndex}.type,'periodic'),edits.(opposite)=struct('type','outflow');end
                p.face_edits=edits;torchDesktopConfig('create',p);app.FaceEdits=edits;app.Status.Text=['Applied ' key ': ' kind];
            catch ME,app.fail(ME);end
        end

        function updateFaceControls(app)
            kind=app.FaceTypeDropDown.Value;primitive=strcmp(kind,'inflow')||(strcmp(kind,'pressure_outlet')&&app.BackflowCheck.Value);
            active={};if primitive,active={'face_density','face_u','face_v','face_w'};end
            if primitive||strcmp(kind,'pressure_outlet'),active{end+1}='face_pressure';end
            if strcmp(kind,'stagnation_inlet'),active=[active,{'face_p0','face_t0','face_r'}];end
            if app.Inputs.reactive.Value && any(strcmp(kind,{'inflow','pressure_outlet','stagnation_inlet'})),active{end+1}='face_lambda';end
            if ~strcmp(app.Inputs.dimension.Value,'3D'),active(strcmp(active,'face_w'))=[];end
            locked=app.Running||~isempty(app.Session);names=fieldnames(app.Inputs);
            for k=1:numel(names),key=names{k};if startsWith(key,'face_'),app.Inputs.(key).Enable=onoff(~locked&&any(strcmp(key,active)));end;end
            app.BackflowCheck.Enable=onoff(~locked&&strcmp(kind,'pressure_outlet'));
        end

        function updateControls(app)
            locked=app.Running||~isempty(app.Session);
            for k=1:numel(app.ConfigControls),app.ConfigControls{k}.Enable=onoff(~locked);end
            if ~locked
                d=str2double(app.Inputs.dimension.Value(1));keys={'nz','lz','oz','w'};
                for k=1:numel(keys),app.Inputs.(keys{k}).Enable=onoff(d==3);end
                uniform=any(strcmp(app.Inputs.preset.Value,{'Uniform','Radial blast','Rotating equilibrium'}));
                keys={'density','pressure','u','v','w','rotation'};for k=1:numel(keys),app.Inputs.(keys{k}).Enable=onoff(uniform);end
                rotation=strcmp(app.Inputs.preset.Value,'Rotating equilibrium');app.Inputs.rotation.Enable=onoff(rotation);
                if rotation,app.Inputs.u.Enable='off';app.Inputs.v.Enable='off';app.Inputs.w.Enable='off';end
                if d~=3,app.Inputs.w.Enable='off';end
                app.Inputs.gamma.Enable=onoff(~app.Inputs.reactive.Value);app.Inputs.fraction.Enable=onoff(app.Inputs.reactive.Value);app.updateFaceControls();
            end
            app.Inputs.t_end.Enable=onoff(~app.Running);app.Inputs.folder.Enable=onoff(~app.Running);app.Inputs.interval.Enable=onoff(~app.Running);
            app.RunButton.Enable=onoff(~app.Running);app.PauseButton.Enable=onoff(app.Running);app.NewButton.Enable=onoff(~app.Running);
            for k=1:2,app.FileButtons{k}.Enable=onoff(~app.Running);end
            app.FileButtons{3}.Enable=onoff(~app.Running&&~isempty(app.Session));
            for k=4:5,app.FileButtons{k}.Enable=onoff(~app.Running&&~isempty(app.Displayed));end
            available=~app.Running&&~isempty(app.Frames);app.FrameSlider.Enable=onoff(available);app.PlayButton.Enable=onoff(available);app.LatestButton.Enable=onoff(available);
        end

        function record(app)
            formats={'xdmf'};if exist([app.Prefix '.series.json'],'file'),data=torchDesktopHistory(app.Prefix);formats=data.formats;if ischar(formats),formats={formats};end;end
            torchWriteSeries(app.Prefix,app.Session,formats);
        end

        function prefix=freshPrefix(app)
            [~,identity]=fileparts(tempname);folder=char(app.Inputs.folder.Value);assert(~isempty(folder),'Choose a results folder');prefix=fullfile(folder,['run_' identity],'history');
        end

        function saveCurrent(app,path)
            assert(~app.Running && ~isempty(app.Session),'Pause before saving a restart');
            torchWriteCheckpoint(path,app.Session);app.Status.Text='Saved current restart.';
        end

        function exportDisplayed(app,path)
            assert(~app.Running && ~isempty(app.Displayed),'Pause before exporting a frame');
            torchDesktopExport(path,app.Displayed);app.Status.Text='Exported displayed frame.';
        end

        function exportFigure(app,path)
            assert(~app.Running && ~isempty(app.Displayed),'Pause before exporting a figure');
            fig=figure('Visible','off','Position',[0,0,1000,800]);cleanup=onCleanup(@()delete(fig));
            ax=axes(fig,'Position',[.10,.40,.72,.53]);profile=axes(fig,'Position',[.10,.10,.80,.18]);
            names=torchCoordinates(app.Displayed.config);normal=find(strcmp(names,app.AxisDropDown.Value));
            torchDesktopDraw(app.Displayed,ax,profile,app.FieldDropDown.Value,normal,round(app.SliceSlider.Value),app.ModeDropDown.Value);
            if endsWith(lower(path),'.svg'),print(fig,path,'-dsvg');else,exportgraphics(fig,path,'Resolution',200);end
            app.Status.Text='Exported displayed figure.';
        end

        function fileAction(app,operation)
            try
                if any(strcmp(operation,{'load','series'}))
                    filter={'*.h5','TORCH restart'};if strcmp(operation,'series'),filter={'*.series.json','TORCH recorded series'};end
                    [name,folder]=uigetfile(filter);if isequal(name,0),return;end;app.load(fullfile(folder,name),strcmp(operation,'series'));
                elseif strcmp(operation,'save')
                    [name,folder]=uiputfile({'*.h5','TORCH restart'});if ~isequal(name,0),app.saveCurrent(fullfile(folder,name));end
                elseif strcmp(operation,'export')
                    [name,folder]=uiputfile({'*.xdmf','XDMF';'*.vtr;*.vts','VTK';'*.csv','CSV';'*.h5','HDF5 snapshot'});if ~isequal(name,0),app.exportDisplayed(fullfile(folder,name));end
                else
                    [name,folder]=uiputfile({'*.png','PNG';'*.svg','SVG'});if isequal(name,0),return;end
                    app.exportFigure(fullfile(folder,name));
                end
            catch ME,app.fail(ME);end
        end

        function fail(app,error)
            if isa(error,'MException'),text=error.message;else,text=char(error);end
            app.Status.Text=['Error: ' text];
        end

        function stopStepTimer(app)
            if ~isempty(app.StepTimer)&&isvalid(app.StepTimer),stop(app.StepTimer);delete(app.StepTimer);end;app.StepTimer=[];
        end

        function releaseBusy(app),app.Busy=false;end
        function timerError(app,event),app.stopStepTimer();app.Running=false;app.fail(event.Data);app.updateControls();end

        function delete(app)
            app.Closing=true;app.Running=false;app.stopStepTimer();app.stopPlayback();
            if ~isempty(app.Figure)&&isvalid(app.Figure),delete(app.Figure);end
        end
    end
    methods (Access=private)
        function buildUI(app,visible)
            app.Figure=uifigure('Name','TORCH | 2D / 3D Simulation Desktop','Position',[80,70,1420,920],'Visible',visible);
            root=uigridlayout(app.Figure,[2,2]);root.RowHeight={'1x',30};root.ColumnWidth={410,'1x'};
            left=uigridlayout(root,[2,1]);left.Layout.Row=1;left.Layout.Column=1;left.RowHeight={'1x',40};
            tabs=uitabgroup(left);tabs.Layout.Row=1;setup=uitab(tabs,'Title','Setup');mesh=uitab(tabs,'Title','Mesh');boundary=uitab(tabs,'Title','Boundaries');files=uitab(tabs,'Title','Files');
            S=uigridlayout(setup,[19,2]);S.RowHeight=repmat({28},1,19);S.ColumnWidth={180,'1x'};
            app.control(S,1,'Dimensions','dimension',{'2D','3D'},'2D');app.control(S,2,'Geometry','geometry',{'cartesian','cylindrical','annular','spherical'},'cartesian');
            app.control(S,3,'Initial condition','preset',{'Uniform'},'Uniform');app.Inputs.dimension.ValueChangedFcn=@(~,~)app.domainChanged();app.Inputs.geometry.ValueChangedFcn=@(~,~)app.domainChanged();app.Inputs.preset.ValueChangedFcn=@(~,~)app.presetChanged();
            app.control(S,4,'End time [s]','t_end',[],.05,false);app.control(S,5,'CFL','cfl',[],.4);app.control(S,6,'Inert gamma','gamma',[],1.4);
            app.control(S,7,'Limiter','limiter',{'VanLeer','Albada','MINBEE','Superbee','Unos'},'VanLeer');
            check=uicheckbox(S,'Text','Reactive flow','Value',false);check.Layout.Row=8;check.Layout.Column=[1,2];check.ValueChangedFcn=@(~,~)app.updateControls();app.Inputs.reactive=check;app.ConfigControls{end+1}=check;
            label=uilabel(S,'Text','MATLAB chemistry: Arrhenius');label.Layout.Row=9;label.Layout.Column=[1,2];
            app.control(S,10,'Initial unburned fraction','fraction',[],1);
            label=uilabel(S,'Text','Uniform / geometric initial data','FontWeight','bold');label.Layout.Row=11;label.Layout.Column=[1,2];
            app.control(S,12,'Density [kg/m^3]','density',[],1);app.control(S,13,'Local velocity 1 [m/s]','u',[],0);app.control(S,14,'Local velocity 2 [m/s]','v',[],0);app.control(S,15,'Local velocity 3 [m/s]','w',[],0);
            app.control(S,16,'Static pressure [Pa]','pressure',[],101325);app.control(S,17,'Rotation rate [rad/s]','rotation',[],20);
            M=uigridlayout(mesh,[16,2]);M.RowHeight=repmat({28},1,16);M.ColumnWidth={180,'1x'};
            for k=1:3,axis='xyz';app.control(M,1+4*(k-1),['Cells ' axis(k)],['n' axis(k)],[],20);app.control(M,2+4*(k-1),['Extent ' axis(k)],['l' axis(k)],[],1);app.control(M,3+4*(k-1),['Origin ' axis(k)],['o' axis(k)],[],0);end
            label=uilabel(M,'Text','Lengths: m. Angles: rad. Cylindrical 2D: r,z. Annular: r,theta. Spherical: r,theta,phi. 3D curved grids require positive inner radius.','WordWrap','on');label.Layout.Row=[13,16];label.Layout.Column=[1,2];
            B=uigridlayout(boundary,[17,2]);B.RowHeight=repmat({28},1,17);B.ColumnWidth={180,'1x'};
            app.FaceDropDown=uidropdown(B,'Items',torchDesktopConfig('keys',2),'ValueChangedFcn',@(~,~)app.showFace());app.FaceDropDown.Layout.Row=1;app.FaceDropDown.Layout.Column=[1,2];app.ConfigControls{end+1}=app.FaceDropDown;
            app.FaceTypeDropDown=uidropdown(B,'Items',{'Outflow','Slip wall','Periodic','Prescribed inflow','Pressure outlet','Reservoir inlet'},'ItemsData',{'outflow','wall','periodic','inflow','pressure_outlet','stagnation_inlet'},'Value','outflow','ValueChangedFcn',@(~,~)app.updateFaceControls());app.FaceTypeDropDown.Layout.Row=2;app.FaceTypeDropDown.Layout.Column=[1,2];app.ConfigControls{end+1}=app.FaceTypeDropDown;
            specs={'Inlet/backflow density','face_density',1;'Local velocity 1 [m/s]','face_u',10;'Local velocity 2 [m/s]','face_v',0;'Local velocity 3 [m/s]','face_w',0;'Static pressure [Pa]','face_pressure',101325;'Total pressure [Pa]','face_p0',110000;'Total temperature [K]','face_t0',300;'Gas constant [J/(kg K)]','face_r',287;'Incoming unburned fraction','face_lambda',1};
            for k=1:size(specs,1),app.control(B,k+2,specs{k,1},specs{k,2},[],specs{k,3});end
            app.BackflowCheck=uicheckbox(B,'Text','Specify outlet backflow','ValueChangedFcn',@(~,~)app.updateFaceControls());app.BackflowCheck.Layout.Row=12;app.BackflowCheck.Layout.Column=[1,2];app.ConfigControls{end+1}=app.BackflowCheck;
            app.ApplyFaceButton=uibutton(B,'Text','Apply face condition','ButtonPushedFcn',@(~,~)app.applyFace());app.ApplyFaceButton.Layout.Row=13;app.ApplyFaceButton.Layout.Column=[1,2];app.ConfigControls{end+1}=app.ApplyFaceButton;
            label=uilabel(B,'Text','Periodic applies to both faces. Local velocities follow the geometry coordinate order. Apply before Run.','WordWrap','on');label.Layout.Row=[14,17];label.Layout.Column=[1,2];
            F=uigridlayout(files,[13,2]);F.RowHeight=repmat({32},1,13);F.ColumnWidth={180,'1x'};
            app.control(F,1,'Results folder','folder','text','out/desktop',false);button=uibutton(F,'Text','Choose folder','ButtonPushedFcn',@(~,~)app.chooseFolder());button.Layout.Row=2;button.Layout.Column=[1,2];
            app.control(F,3,'Record every N steps','interval',[],5,false);
            titles={'Load restart','Open recorded series','Save current restart','Export displayed data','Export displayed figure'};operations={'load','series','save','export','figure'};
            for k=1:numel(titles),operation=operations{k};button=uibutton(F,'Text',titles{k},'ButtonPushedFcn',@(~,~)app.fileAction(operation));button.Layout.Row=k+3;button.Layout.Column=[1,2];app.FileButtons{k}=button;end
            label=uilabel(F,'Text','Frames stay on disk. Save current restart uses the latest simulation state. Displayed exports may use an earlier frame; Resume always continues the latest state.','WordWrap','on');label.Layout.Row=[9,13];label.Layout.Column=[1,2];
            actions=uigridlayout(left,[1,3]);actions.Layout.Row=2;app.RunButton=uibutton(actions,'Text','Run / resume','ButtonPushedFcn',@(~,~)app.run());app.PauseButton=uibutton(actions,'Text','Pause','ButtonPushedFcn',@(~,~)app.pause());app.NewButton=uibutton(actions,'Text','New','ButtonPushedFcn',@(~,~)app.reset());
            right=uigridlayout(root,[6,1]);right.Layout.Row=1;right.Layout.Column=2;right.RowHeight={45,'3x','1x',40,25,25};
            controls=uigridlayout(right,[1,4]);controls.Layout.Row=1;controls.ColumnWidth={160,105,80,'1x'};
            app.FieldDropDown=uidropdown(controls,'Items',{'Pressure'},'ValueChangedFcn',@(~,~)app.redraw());app.ModeDropDown=uidropdown(controls,'Items',{'Map','3D plane'},'ValueChangedFcn',@(~,~)app.redraw());app.AxisDropDown=uidropdown(controls,'Items',{'x','y','z'},'Value','z','ValueChangedFcn',@(~,~)app.updateSlice(true));
            app.SliceSlider=uislider(controls,'MinorTicks',[],'Limits',[1,20],'Value',1,'ValueChangedFcn',@(~,~)app.redraw(),'Tooltip','Slice index (one-based)');plotPanel=uipanel(right,'BorderType','none');plotPanel.Layout.Row=2;profilePanel=uipanel(right,'BorderType','none');profilePanel.Layout.Row=3;
            app.PlotAxes=uiaxes(plotPanel,'Units','normalized','Position',[.08,.12,.78,.82]);
            app.ProfileAxes=uiaxes(profilePanel,'Units','normalized','Position',[.08,.22,.88,.65]);
            timeline=uigridlayout(right,[1,3]);timeline.Layout.Row=4;timeline.ColumnWidth={110,80,'1x'};app.PlayButton=uibutton(timeline,'Text','Play','ButtonPushedFcn',@(~,~)app.play());app.LatestButton=uibutton(timeline,'Text','Latest','ButtonPushedFcn',@(~,~)app.renderHistory(numel(app.Frames)));
            app.FrameSlider=uislider(timeline,'MinorTicks',[],'Limits',[1,2],'Value',1,'ValueChangedFcn',@(~,~)app.renderHistory(app.FrameSlider.Value));app.ViewTime=uilabel(right,'Text','No recorded frames');app.ViewTime.Layout.Row=5;app.Audit=uilabel(right,'Text','No results yet');app.Audit.Layout.Row=6;
            app.Status=uilabel(root,'Text','Configure a simulation, then Run.');app.Status.Layout.Row=2;app.Status.Layout.Column=[1,2];
        end

        function control(app,parent,row,label,key,items,value,configuration)
            if nargin<8,configuration=true;end
            text=uilabel(parent,'Text',label);text.Layout.Row=row;text.Layout.Column=1;app.Labels.(key)=text;
            if iscell(items),widget=uidropdown(parent,'Items',items,'Value',value);
            elseif ischar(items)&&strcmp(items,'text'),widget=uieditfield(parent,'text','Value',value);
            else,widget=uieditfield(parent,'numeric','Value',value);end
            widget.Layout.Row=row;widget.Layout.Column=2;app.Inputs.(key)=widget;if configuration,app.ConfigControls{end+1}=widget;end
        end

        function domainChanged(app)
            if app.Adjusting,return;end;app.configure(torchDesktopConfig('defaults',str2double(app.Inputs.dimension.Value(1)),app.Inputs.geometry.Value));
        end

        function presetChanged(app)
            app.FaceEdits=struct;reactive=any(strcmp(app.Inputs.preset.Value,{'2d_rde_chamber','3d_ignition'}));app.Inputs.reactive.Value=reactive;
            tend=.05;if ~strcmp(app.Inputs.geometry.Value,'cartesian'),tend=1e-4;end;if reactive,tend=1e-5;end
            app.Inputs.t_end.Value=tend;app.updateControls();app.showFace();
        end

        function chooseFolder(app)
            if app.Running,return;end;folder=uigetdir();if ~isequal(folder,0),app.Inputs.folder.Value=folder;end
        end
    end
end

function value=onoff(condition)
value='off';if condition,value='on';end
end
