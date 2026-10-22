function [U,t,lambda,X,Y,Z,diagnostic] = inputLoopResult3D(varargin)
% Name/value wrapper around torchSolver3D; returns final state only.
result=torchSolver3D(varargin{:});
U=result.u_final; t=result.time; lambda=result.lambda_final;
X=result.X; Y=result.Y; Z=result.Z;
diagnostic=rmfield(result,{'u_final','u_prim_final','lambda_final','X','Y','Z','x','y','z'});
end
