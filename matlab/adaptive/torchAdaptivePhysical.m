function yes = torchAdaptivePhysical(U)
yes=all(isfinite(U(:)))&&all(U(1,:)>1e-12)&&all(U(end,:)-.5*sum(U(2:end-1,:).^2,1)./U(1,:)>1e-12);
end
