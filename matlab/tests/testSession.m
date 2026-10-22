function testSession
% HDF5 restart and output contracts. Requires MATLAB's native HDF5 support.
addpath(fileparts(fileparts(mfilename('fullpath'))));
setupTorch;
folder=tempname;mkdir(folder);cleanup=onCleanup(@()rmdir(folder,'s'));
for dim=1:3
    for reactive=[false,true]
        counts=[12,6,4];counts=counts(1:dim);shape=counts(end:-1:1);
        P=ones([dim+2,shape]);indices=repmat({':'},1,dim+1);indices{1}=2:dim+1;P(indices{:})=.2;
        t_end=.08;
        if reactive,indices{1}=dim+2;P(indices{:})=1e6;t_end=1e-4;end
        s=torchCreateSession(P,ones(1,dim),'t_end',t_end,'reactive',reactive,'boundaries',repmat({'Periodic'},1,dim));
        full=torchRunSession(s);
        checkpoint=fullfile(folder,'checkpoint.h5');prefix=fullfile(folder,'history');
        s=torchRunSession(s,'step_limit',2,'checkpoint',checkpoint,'checkpoint_every',1,'output_prefix',prefix,'output_every',2);
        assert(s.time<t_end,'Pause did not leave a resumable state');
        resumed=torchReadCheckpoint(checkpoint);resumed=torchRunSession(resumed,'output_prefix',prefix,'output_every',2);
        assert(isequal(resumed.state,full.state),'Restart changed conservative state');
        assert(isequal(resumed.lam,full.lam),'Restart changed composition');
        assert(isequal(resumed.boundary_integral,full.boundary_integral) && isequal(resumed.source_integral,full.source_integral),'Restart changed balances');
        assert(resumed.time==full.time && resumed.steps==full.steps,'Restart changed integration sequence');
        torchWriteVTK(fullfile(folder,'state.vtr'),resumed);
        torchWriteXDMF(fullfile(folder,'state.xdmf'),resumed);
        snapshot=torchReadCheckpoint(fullfile(folder,'state.h5'));assert(isequal(snapshot.state,full.state),'Snapshot not restartable');
        manifest=jsondecode(fileread([prefix '.series.json']));assert(manifest.frames(end).step==full.steps,'Temporal series lacks final state');
        for k=2:numel(manifest.frames),assert(manifest.frames(k).time>manifest.frames(k-1).time,'Repeated/nonmonotone frame time');end
        rejected=false;try,torchWriteSeries(prefix,torchCreateSession(P,ones(1,dim),'t_end',t_end));catch,rejected=true;end
        assert(rejected,'Series overwrite by another run accepted');
        delete([prefix '.series.json']);delete([prefix '.pvd']);delete([prefix '.xdmf']);
    end
end
% Unknown format version and corrupt state must fail before integration.
s=torchCreateSession([ones(1,8);zeros(1,8);ones(1,8)],1,'t_end',.01);
path=fullfile(folder,'invalid.h5');torchWriteCheckpoint(path,s);h5write(path,'/state/rho',zeros(8,1));
rejected=false;try,torchReadCheckpoint(path);catch,rejected=true;end;assert(rejected,'Corrupt checkpoint accepted');
fprintf('ALL MATLAB CHECKPOINT, RESTART AND TEMPORAL OUTPUT CHECKS PASSED!\n');
end
