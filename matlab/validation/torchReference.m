function U = torchReference(kind,n,time,dimension)
% Independent analytic Euler reference averages; no numerical solver calls.
% Vortex: NASA 20150018403. Shock: NASA Glenn / NACA Report 1135.
% Acoustics: right-running linear Euler mode, finite amplitude error O(1e-12).
if nargin<3,time=0;end
if nargin<4,dimension=2;end
assert(isscalar(n)&&isfinite(n)&&n>=2&&n==fix(n),'Invalid reference mesh');
assert(any(dimension==1:3)&&isscalar(time)&&isfinite(time),'Invalid reference dimension/time');
if strcmp(kind,'shock')
    mach=2;gamma=1.4;ratio=(gamma+1)*mach^2/((gamma-1)*mach^2+2);
    upstream=[1;mach*sqrt(gamma);1];downstream=[ratio;upstream(2)/ratio;(2*gamma*mach^2-(gamma-1))/(gamma+1)];
    counts=[n,repmat(4,1,dimension-1)];shape=counts(end:-1:1);if dimension==1,shape=[1,n];end
    x=((1:n)-.5)/n;values=zeros(dimension+2,prod(counts));
    mask=repmat(x<.5,prod(counts(2:end)),1);mask=mask(:)';
    rows=[1,2,dimension+2];
    for k=1:3,values(rows(k),:)=downstream(k)+(upstream(k)-downstream(k))*mask;end
    U=reshape(conservative(values),[dimension+2,counts(end:-1:1)]);return;
end
if strcmp(kind,'vortex')
    assert(dimension>=2,'Vortex requires 2D or 3D');h=10/n;x=-5+((1:n)-.5)*h;[Y,X]=ndgrid(x,x);U=zeros(dimension+2,n,n);
    for sx=[-1,1]
        for sy=[-1,1]
            xx=mod(X+sx*h/sqrt(12)-time+5,10)-5;yy=mod(Y+sy*h/sqrt(12)-time+5,10)-5;r2=xx.^2+yy.^2;
            swirl=5/(2*pi)*exp(.5*(1-r2));temperature=1-.4*25/(8*1.4*pi^2)*exp(1-r2);
            P=zeros(dimension+2,n*n);P(1,:)=reshape(temperature.^2.5,1,[]);P(2,:)=reshape(1-yy.*swirl,1,[]);P(3,:)=reshape(1+xx.*swirl,1,[]);
            if dimension==3,P(4,:)=.25;end
            P(end,:)=P(1,:).^1.4;U=U+.25*reshape(conservative(P),size(U));
        end
    end
    if dimension==3,U=repmat(reshape(U,[5,1,n,n]),[1,4,1,1]);end
elseif strcmp(kind,'acoustic')
    shape=repmat(n,1,dimension);if dimension==1,shape=[1,n];end
    centers=((1:n)-.5)/n;coordinates=cell(1,dimension);
    if dimension==1,coordinates{1}=centers;else,[coordinates{:}]=ndgrid(centers);end
    phase=zeros(shape);for k=1:dimension,phase=phase+coordinates{k};end
    phase=2*pi*(phase-(.2*dimension+sqrt(1.4*dimension))*time);
    wave=1e-6*sin(phase)*(sin(pi/n)/(pi/n))^dimension;wave=wave(:)';U=zeros(dimension+2,n^dimension);
    U(1,:)=1+wave;normal=sqrt(1.4/dimension);
    for k=1:dimension,U(k+1,:)=.2+(.2+normal)*wave;end
    U(end,:)=2.5+.5*dimension*.2^2+(1.4/.4+.5*dimension*.2^2+dimension*.2*normal)*wave;
    U=reshape(U,[dimension+2,repmat(n,1,dimension)]);
else,error('Unknown analytic reference');
end
end

function U=conservative(P)
U=P;U(2:end-1,:)=P(1,:).*P(2:end-1,:);U(end,:)=P(end,:)/.4+.5*P(1,:).*sum(P(2:end-1,:).^2,1);
end
