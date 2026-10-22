function testPerformance
% Bounded HLLC batches, integer index memory and cache correctness.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
for nv=[4,5]
    P=ones(nv,5,7);P(2:end-1,:,:)=.2;P(end,:,:)=2;flat=reshape(P,nv,[]);
    U=flat;U(2:end-1,:)=flat(1,:).*flat(2:end-1,:);U(end,:)=flat(end,:)/.4+.5*flat(1,:).*sum(flat(2:end-1,:).^2,1);U=reshape(U,size(P));
    for direction=1:nv-2
        expected=.2*U;expected(direction+1,:,:)=expected(direction+1,:,:)+2;expected(end,:,:)=expected(end,:,:)+.4;
        for batch=[1,13,8192]
            actual=torchHllcGrid(U,U,1.4,1.4,direction,batch);assert(max(abs(actual(:)-expected(:)))<1e-14,'Batched physical flux');
        end
    end
end
P=ones(5,4,6,8);P(2:4,:,:,:)=.1;s=torchCreateSession(P,[.5,1,1],'geometry','spherical','origins',[.5,.3,.1],'t_end',.01);c=s.config;
clear torchSessionGrid;
G=torchSessionGrid(c);assert(isa(G.index{1}.left,'uint32'),'Metric indices are not compact');
assert(isequal(G,torchMetricGrid(c)),'Cached grid differs from direct metrics');
original=G.volume;G.volume(1)=123;again=torchSessionGrid(c);assert(isequal(again.volume,original),'Returned grid mutated the cache');
for action=1:5
    edited=c;
    if action==1,edited.origins_xyz(1)=.6;
    elseif action==2,edited.lengths_xyz(1)=.6;
    elseif action==3,edited.counts_xyz(1)=10;
    elseif action==4,edited.geometry='annular';
    else,edited.geometry='cartesian';edited.face_boundaries{1}=struct('type','periodic');edited.face_boundaries{2}=struct('type','periodic');end
    assert(isequal(torchSessionGrid(edited),torchMetricGrid(edited)),'Cache did not invalidate');
    assert(isequal(torchSessionGrid(c),torchMetricGrid(c)),'Cache did not return to original mesh');
end
clear torchSessionGrid;
fprintf('PASS MATLAB/Octave HLLC batches, compact metric indices and cache invalidation\n');
end
