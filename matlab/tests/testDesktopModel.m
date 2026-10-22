function testDesktopModel
% Pure model and slice checks also run in Octave without a graphical display.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
for d=[2,3]
    for item={'cartesian','cylindrical','annular','spherical'}
        geometry=item{1};p=torchDesktopConfig('defaults',d,geometry);p.counts=[8,6,4];p.counts=p.counts(1:d);
        choices=torchDesktopConfig('presets',d,geometry);
        for k=1:numel(choices)
            p.preset=choices{k};s=torchDesktopConfig('create',p);
            assert(isequal(size(s.state),[d+2,p.counts(end:-1:1)]),'Invalid desktop mesh');
            plane=torchDesktopPlane(s,'Pressure',d,2);assert(all(isfinite(plane.values(:))) && all(plane.values(:)>0),'Nonphysical desktop pressure');
        end
    end
end
x=1+((1:7)-.5)*2/7;y=2+((1:5)-.5)*3/5;z=3+((1:3)-.5)*4/3;[Z,Y,X]=ndgrid(z,y,x);
P=ones(5,3,5,7);P(2:4,:,:,:)=0;P(5,:,:,:)=reshape(10+X+10*Y+100*Z,1,3,5,7);
s=torchCreateSession(P,[2,3,4],'origins',[1,2,3],'t_end',.1);
for normal=1:3
    plane=torchDesktopPlane(s,'Pressure',normal,2);pick={':',':',':'};pick{4-normal}=2;expected=squeeze(10+X+10*Y+100*Z);expected=squeeze(expected(pick{:}));
    assert(max(abs(plane.values(:)-expected(:)))<1e-12,'Incorrect desktop slice orientation');
    value={x,y,z};position=plane.vertices(:,:,normal);assert(max(abs(position(:)-value{normal}(2)))<1e-12,'Slice not at cell centre');
end
fprintf('PASS MATLAB/Octave desktop presets, geometries and slice orientation\n');
end
