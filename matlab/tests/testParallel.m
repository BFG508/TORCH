function testParallel()
% Native PCT execution must preserve each serial numerical kernel exactly.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
pool=torchWorkers(2);assert(~isempty(pool)&&pool.NumWorkers>=2,'No actual parallel pool');
for kind=1:5
    if kind==1,counts=17000;d=1;elseif kind==2,counts=[96,96];d=2;else,counts=[24,24,24];d=3;end
    P=ones([d+2,fliplr(counts)]);flat=reshape(P,d+2,[]);flat(2:end-1,:)=.1;flat(1,:)=1+.03*sin(1:size(flat,2));P=reshape(flat,size(P));
    geometry='cartesian';origins=zeros(1,d);faces=[];adaptive=[];
    if kind==4,geometry='spherical';origins=[.5,.3,.2];faces=struct;end
    if kind==5,adaptive=struct('max_level',1,'regrid_every',1,'max_cells',50000);end
    serial=torchCreateSession(P,ones(1,d),'t_end',.003,'geometry',geometry,'origins',origins,'face_boundaries',faces,'adaptive',adaptive);
    parallel=serial;parallel.workers=2;serial=torchRunSession(serial,'step_limit',2);parallel=torchRunSession(parallel,'step_limit',2);
    assert(isequal(serial.state,parallel.state),'Parallel state differs from serial');
    assert(isequal(serial.boundary_integral,parallel.boundary_integral),'Parallel balance differs');
    if isfield(serial,'mesh'),assert(isequal(serial.mesh.keys,parallel.mesh.keys),'Parallel regrid differs');end
end
assert(isequal(gcp('nocreate'),pool),'TORCH replaced an existing user pool');
path=[tempname '.h5'];cleanup=onCleanup(@()delete(path));torchWriteCheckpoint(path,parallel);r=torchReadCheckpoint(path);assert(r.workers==1,'Host execution leaked into checkpoint');
fprintf('PASS MATLAB serial/parallel equality in all dimensions and kernels\n');
end
