function s = torchVerificationCase(kind,n,dimension,kernel)
% Initialize solver sessions from independent conservative cell averages.
assert(any(strcmp(kernel,{'hllc','metric'})),'Choose hllc or metric');
U=torchReference(kind,n,0,dimension);shape=size(U);P=reshape(U,dimension+2,[]);rho=P(1,:);
P(2:end-1,:)=P(2:end-1,:)./rho;P(end,:)=.4*(P(end,:)-.5*rho.*sum(P(2:end-1,:).^2,1));P=reshape(P,shape);
L=ones(1,dimension);origins=zeros(1,dimension);target=.05;boundaries=repmat({'Periodic'},1,dimension);
if strcmp(kind,'vortex'),L=[10,10];origins=[-5,-5];target=.5;if dimension==3,L(3)=1;origins(3)=0;end
elseif strcmp(kind,'shock'),boundaries{1}='Open';end
options={'t_end',target,'boundaries',boundaries,'origins',origins};
if strcmp(kernel,'metric')
    faces=struct;
    for k=1:dimension,axis='xyz';faces.([axis(k) '_min'])='periodic';faces.([axis(k) '_max'])='periodic';end
    if strcmp(kind,'shock')
        pressure=(2*1.4*4-.4)/2.4;
        faces.x_min=struct('type','inflow','primitive',[1,2*sqrt(1.4),zeros(1,dimension-1),1]);
        faces.x_max=struct('type','pressure_outlet','pressure',pressure);
    end
    options=[options,{'face_boundaries',faces}];
end
s=torchCreateSession(P,L,options{:});
end
