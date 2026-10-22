function s = torchReadCheckpoint(path,varargin)
% Read Python or MATLAB schema-1/2 checkpoint; reject unsupported versions/models.
p=inputParser;addParameter(p,'t_end',[]);parse(p,varargin{:});
try
    bytes=h5read(path,'/metadata');assert(isa(bytes,'uint8') && numel(bytes)<1e6,'Invalid metadata');
    meta=jsondecode(native2unicode(bytes(:)','UTF-8'));
    if meta.schema_version==3
        s=torchReadAdaptive(path,meta);
        if ~isempty(p.Results.t_end),assert(isfinite(p.Results.t_end)&&p.Results.t_end>=s.time,'New end time precedes checkpoint');s.config.t_end=p.Results.t_end;end
        return;
    end
    assert(strcmp(meta.format,'TORCH checkpoint') && ((meta.schema_version==1 && strcmp(meta.solver_version,'1.1')) || (meta.schema_version==2 && any(strcmp(meta.solver_version,{'1.2','1.4'})))) && strcmp(meta.units,'SI') && strcmp(meta.layout,'x-fastest'),'Unsupported checkpoint schema, geometry or units');
    c=meta.config;geometry='cartesian';origins=zeros(1,c.dimension);faces=[];
    if isfield(c,'geometry'),geometry=c.geometry;end
    if isfield(c,'origins_xyz'),origins=c.origins_xyz(:)';c.origins_xyz=origins;end
    assert(strcmp(meta.geometry,geometry),'Checkpoint geometry mismatch');
    if meta.schema_version==1,assert(strcmp(geometry,'cartesian') && ~isfield(c,'geometry') && ~isfield(c,'origins_xyz') && ~isfield(c,'face_boundaries'),'Schema 1 only supports legacy Cartesian grids');end
    if isfield(c,'face_boundaries'),faces=c.face_boundaries;end
    c.counts_xyz=c.counts_xyz(:)';c.lengths_xyz=c.lengths_xyz(:)';
    if ischar(c.boundaries_xyz),c.boundaries_xyz={c.boundaries_xyz};end
    c.boundaries_xyz=c.boundaries_xyz(:)';
    assert(~c.reactive || strcmp(c.kinetic_model,'arrhenius'),'MATLAB restart of ZND is unsupported');
    dim=c.dimension;nvars=dim+2;shape=c.counts_xyz(end:-1:1);
    if dim==1,fullshape=[nvars,shape];else,fullshape=[nvars,shape];end
    axes=torchCoordinates(c);names=[{'rho'},cellfun(@(a)['momentum_' a],axes,'UniformOutput',false),{'energy'}];
    U=zeros(fullshape);flat=zeros(nvars,prod(shape));
    for k=1:nvars
        location=['/state/' names{k}];info=h5info(path,location);
        assert(numel(info.Dataspace.Size)==1 && info.Dataspace.Size==prod(shape),'Invalid dataset rank');
        flat(k,:)=restore(h5read(path,location),c);
    end
    U=reshape(flat,fullshape);lam=[];beta=[];
    if c.reactive,lam=restore(h5read(path,'/state/lambda'),c);if dim==1,lam=reshape(lam,1,[]);else,lam=reshape(lam,shape);end;end
    gamma=c.gamma;if c.reactive,gamma=(4.5-lam)./(3.5-lam);gamma=gamma(:)';end
    P=flat;P(2:end-1,:)=flat(2:end-1,:)./flat(1,:);P(end,:)=(gamma-1).*(flat(end,:)-.5*sum(flat(2:end-1,:).^2,1)./flat(1,:));
    % Reuse constructor validation without reinitializing the conservative state.
    s=torchCreateSession(reshape(P,fullshape),c.lengths_xyz,'t_end',c.t_end,'cfl',c.cfl,'gamma',c.gamma,...
        'boundaries',c.boundaries_xyz,'limiter',c.limiter,'reactive',c.reactive,'lambda_init',lam,...
        'inlet_primitive',c.inlet_primitive,'controller',c.controller,'variant',c.variant,'wavespeed',c.wavespeed,'max_steps',c.max_steps,'geometry',geometry,'origins',origins,'face_boundaries',faces);
    if isfield(c,'face_boundaries'),c.face_boundaries=s.config.face_boundaries;end
    assert(isfinite(meta.time) && meta.time>=0 && meta.time<=c.t_end && meta.steps>=0 && meta.steps==fix(meta.steps) && meta.steps<=c.max_steps,'Invalid time or step metadata');
    s.state=U;current=torchSessionIntegral(s);
    assert(all(abs(current-meta.current_integral(:))<=1e-12+1e-12*abs(current)),'State/integral mismatch');
    for key={'initial_integral','boundary_integral','source_integral'}
        value=meta.(key{1});assert(numel(value)==nvars && all(isfinite(value(:))),'Invalid balances');s.(key{1})=value(:);
    end
    assert(ischar(meta.run_id) && ~isempty(meta.run_id) && s.initial_integral(1)>0,'Invalid run identity');
    s.state=U;s.lam=lam;s.beta=beta;s.config=c;s.time=meta.time;s.steps=meta.steps;s.run_id=meta.run_id;
    info=h5info(path,'/state');expected=names;if c.reactive,expected{end+1}='lambda';end
    assert(isequal(sort({info.Datasets.Name}),sort(expected)),'Unexpected checkpoint fields');
catch ME
    error('TORCH:Checkpoint','Invalid or unsupported TORCH checkpoint: %s',ME.message);
end
if ~isempty(p.Results.t_end)
    assert(isfinite(p.Results.t_end) && p.Results.t_end>=s.time,'New end time precedes checkpoint');s.config.t_end=p.Results.t_end;
end
end

function row=restore(vector,c)
assert(isa(vector,'double') && numel(vector)==prod(c.counts_xyz) && all(isfinite(vector(:))),'Invalid state dataset');
if c.dimension==1,row=vector(:)';
else
    A=reshape(vector,c.counts_xyz);A=permute(A,c.dimension:-1:1);row=A(:)';
end
end
