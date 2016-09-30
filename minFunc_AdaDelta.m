function [paraVec, funcVal, funcVec] = ...
  minFunc_AdaDelta(funcHndl, paraVec, opts, varargin)
% INTRO
%   minimize the objective function via AdaDelta
% INPUT
%   funcHndl: function handler (compute the function's value and gradient)
%   paraVec: D x 1 (initial solution)
%   opts: structure (optimization options)
%   varargin: K x 1 (cell array; additional parameters)
% OUTPUT
%   paraVec: D x 1 (optimal solution)
%   funcVal: scalar (function's value of the optimal solution)
%   funcVec: T x 1 (list of function's values through iterations)

% solve the optimization via gradient-based update
gradVecAccmPri = zeros(size(paraVec));
gradVecAccmSec = zeros(size(paraVec));
funcVec = zeros(opts.epchCnt + 1, 1);
[funcVec(1), ~] = funcHndl(paraVec, [], varargin{:});
for epchIdx = 1 : opts.epchCnt
  % generate the mini-batch partition
  smplIdxLst = GnrtMiniBatc(opts.smplCnt, opts.batcSiz);
  
  % update parameters with mini-batches
  for batcIdx = 1 : numel(smplIdxLst)
    % obtain the function's value and gradient vector of the current solution
    [~, gradVec] = funcHndl(paraVec, smplIdxLst{batcIdx}, varargin{:});

    % compute the adjusted gradient vector
    gradVecAccmPri = ...
      opts.momentum * gradVecAccmPri + (1 - opts.momentum) * gradVec .^ 2;
    gradVecAdjs = gradVec .* sqrt(...
      gradVecAccmSec + opts.fudgFctr) ./ sqrt(gradVecAccmPri + opts.fudgFctr);
    gradVecAccmSec = ...
      opts.momentum * gradVecAccmSec + (1 - opts.momentum) * gradVecAdjs .^ 2;

    % use gradient to update the solution
    paraVec = paraVec - gradVecAdjs;
  end

  % record related variables
  [funcVal, ~] = funcHndl(paraVec, [], varargin{:});
  funcVec(epchIdx + 1) = funcVal;
end

end
