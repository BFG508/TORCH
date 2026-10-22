function out = torchDesktopConfig(action,varargin)
% Widget-independent desktop defaults, presets, face conditions and sessions.
switch action
    case 'defaults'
        d=varargin{1};g='cartesian';if numel(varargin)>1,g=varargin{2};end
        L=ones(1,d);o=zeros(1,d);
        if strcmp(g,'annular') || (d==3 && ~strcmp(g,'cartesian')),L(1)=.5;o(1)=.5;end
        if (strcmp(g,'annular')&&d>=2)||(strcmp(g,'cylindrical')&&d==3),L(2)=2*pi;end
        if strcmp(g,'spherical')&&d>=2
            L(2)=pi;if d==3,L(2)=pi-.4;o(2)=.2;L(3)=2*pi;end
        end
        counts=[40,32];name='2d_explosion';if d==3,counts=[20,16,12];name='3d_explosion';end
        tend=.05;if ~strcmp(g,'cartesian'),name='Uniform';tend=1e-4;end
        out=struct('dimension',d,'geometry',g,'counts',counts,'lengths',L,'origins',o,'preset',name,...
            't_end',tend,'cfl',.4,'gamma',1.4,'limiter','VanLeer','reactive',false,'density',1,...
            'pressure',101325,'velocities',zeros(1,d),'rotation',20,'fraction',1,'face_edits',struct,'workers',1,'adaptive',false,'amr',torchAdaptiveOptions());
    case 'presets'
        d=varargin{1};g=varargin{2};
        if strcmp(g,'cartesian')
            if d==2,out={'Uniform','2d_explosion','2d_implosion','2d_riemann_quadrant','2d_shock_tube','2d_rde_chamber'};
            else,out={'Uniform','3d_sod_x','3d_sod_y','3d_sod_z','3d_explosion','3d_ignition'};end
        else
            out={'Uniform','Radial blast'};
            if (strcmp(g,'annular')&&d>=2)||d==3,out{end+1}='Rotating equilibrium';end
        end
    case 'keys'
        d=varargin{1};out={'x_min','x_max','y_min','y_max','z_min','z_max'};out=out(1:2*d);
    case 'faces'
        p=varargin{1};d=p.dimension;bc=defaultBoundaries(p);
        c=struct('dimension',d,'boundaries_xyz',{bc},'inlet_primitive',[1,0,100,101325]);
        out=torchBoundaryFaces(c);g=p.geometry;
        if ~strcmp(g,'cartesian')
            if p.origins(1)==0,out{1}=struct('type','wall');end
            if strcmp(g,'spherical') && d>=2
                if p.origins(2)==0,out{3}=struct('type','wall');end
                if abs(p.origins(2)+p.lengths(2)-pi)<1e-12,out{4}=struct('type','wall');end
            end
            az=0;if strcmp(g,'spherical')&&d==3,az=3;elseif (strcmp(g,'annular')&&d>=2)||(strcmp(g,'cylindrical')&&d==3),az=2;end
            if az>0 && abs(p.lengths(az)-2*pi)<1e-12,out(2*az-1:2*az)={struct('type','periodic'),struct('type','periodic')};end
        end
        names=torchDesktopConfig('keys',d);
        for k=1:numel(names),if isfield(p.face_edits,names{k}),out{k}=p.face_edits.(names{k});end;end
    case 'create'
        p=varargin{1};d=p.dimension;n=p.counts;shape=n(end:-1:1);
        assert(any(d==[2,3]) && numel(n)==d && all(isfinite(n)) && all(n>=2 & n==fix(n)),'At least two integer cells per axis required');
        assert(any(strcmp(p.preset,torchDesktopConfig('presets',d,p.geometry))),'Preset does not match grid');
        name=p.preset;bc=defaultBoundaries(p);
        if any(strcmp(name,{'Uniform','Radial blast','Rotating equilibrium'}))
            P=ones([d+2,shape]);index=repmat({':'},1,d+1);index{1}=1;P(index{:})=p.density;index{1}=d+2;P(index{:})=p.pressure;
            assert(numel(p.velocities)==d,'Velocity dimension mismatch');
            for k=1:d,index{1}=k+1;P(index{:})=p.velocities(k);end
            r=p.origins(1)+((1:n(1))-.5)*p.lengths(1)/n(1);
            if strcmp(name,'Radial blast')
                pressure=p.pressure*(1+9*(r<p.origins(1)+.25*p.lengths(1)));
                pressure=repmat(pressure,prod(shape(1:end-1)),1);index{1}=d+2;P(index{:})=reshape(pressure,[1,shape]);
            elseif strcmp(name,'Rotating equilibrium')
                speed=p.rotation*r;direction=3;index{1}=2:d+1;P(index{:})=0;
                if strcmp(p.geometry,'spherical')
                    theta=p.origins(2)+((1:n(2))-.5)*p.lengths(2)/n(2);speed=sin(theta(:))*speed;direction=4;
                    speed=repmat(reshape(speed,[1,n(2),n(1)]),n(3),1,1);
                elseif d==3,speed=repmat(reshape(speed,[1,1,n(1)]),n(3),n(2),1);
                else,speed=repmat(speed,n(2),1);end
                index{1}=direction;P(index{:})=reshape(speed,[1,shape]);
                index{1}=d+2;P(index{:})=reshape(p.pressure+.5*p.density*speed.^2,[1,shape]);
            end
        elseif d==2
            U=torchSolver2D('preset',name,n(1),n(2),0,.4,'VanLeer');P=uconsvToUprim2D(U,1.4);
        else
            initial=torchSolver3D('preset_name',name,'nx',n(1),'ny',n(2),'nz',n(3),'t_end',0);P=initial.u_prim_final;
        end
        lam=[];if p.reactive,lam=p.fraction*ones(shape);end
        faces=[];if ~strcmp(p.geometry,'cartesian') || ~isempty(fieldnames(p.face_edits)),faces=torchDesktopConfig('faces',p);end
        out=torchCreateSession(P,p.lengths,'t_end',p.t_end,'cfl',p.cfl,'gamma',p.gamma,'limiter',p.limiter,...
            'reactive',p.reactive,'lambda_init',lam,'geometry',p.geometry,'origins',p.origins,'boundaries',bc,'face_boundaries',faces,'workers',p.workers);
        if p.adaptive,out=torchAdaptiveSession(out,p.amr);end
    case 'fromSession'
        s=varargin{1};c=s.config;g='cartesian';if isfield(c,'geometry'),g=c.geometry;end
        out=torchDesktopConfig('defaults',max(2,c.dimension),g);out.dimension=c.dimension;
        out.counts=c.counts_xyz;out.lengths=c.lengths_xyz;out.origins=zeros(1,c.dimension);if isfield(c,'origins_xyz'),out.origins=c.origins_xyz;end
        out.t_end=c.t_end;out.cfl=c.cfl;out.gamma=c.gamma;out.limiter=c.limiter;out.reactive=c.reactive;out.preset='Uniform';if isfield(s,'workers'),out.workers=s.workers;end
        out.adaptive=isfield(s,'mesh');if out.adaptive,out.amr=c.amr;end
        if isfield(c,'face_boundaries'),faces=torchBoundaryFaces(c,c.face_boundaries);else,faces=torchBoundaryFaces(c);end
        keys=torchDesktopConfig('keys',c.dimension);for k=1:numel(keys),out.face_edits.(keys{k})=faces{k};end
    otherwise,error('Unknown desktop configuration operation');
end
end

function bc=defaultBoundaries(p)
bc=repmat({'Open'},1,p.dimension);
if strcmp(p.geometry,'cartesian') && p.dimension==2
    if strcmp(p.preset,'2d_implosion'),bc={'Reflective','Reflective'};
    elseif strcmp(p.preset,'2d_shock_tube'),bc={'Open','Reflective'};
    elseif strcmp(p.preset,'2d_rde_chamber'),bc={'Periodic','Inflow'};end
end
end
