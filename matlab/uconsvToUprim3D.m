function P = uconsvToUprim3D(U, gamma)
% Primitive layout [5,nz,ny,nx]: rho,u,v,w,p; SI units.
if nargin<2, gamma=1.4; end
if ~isscalar(gamma), gamma=reshape(gamma,[1,size(U,2),size(U,3),size(U,4)]); end
P = double(U);
P(2:4,:,:,:) = U(2:4,:,:,:)./U(1,:,:,:);
P(5,:,:,:) = (gamma-1).*(U(5,:,:,:)-0.5.*sum(U(2:4,:,:,:).^2,1)./U(1,:,:,:));
end
