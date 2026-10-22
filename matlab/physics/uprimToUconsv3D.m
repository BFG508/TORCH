function U = uprimToUconsv3D(P, gamma)
% State layout [5,nz,ny,nx]: rho,rho*u,rho*v,rho*w,E; SI units.
if nargin<2, gamma=1.4; end
if ~isscalar(gamma), gamma=reshape(gamma,[1,size(P,2),size(P,3),size(P,4)]); end
U = double(P);
U(2:4,:,:,:) = P(2:4,:,:,:) .* P(1,:,:,:);
U(5,:,:,:) = P(5,:,:,:)./(gamma-1) + 0.5.*P(1,:,:,:).*sum(P(2:4,:,:,:).^2,1);
end
