function [funcVal, funcValLst, paraVec] = ...
  minFunc_Adam(paraVec, funcHndl, opts, varargin)
% INTRO
%   minimize the objective function via Adam
% INPUT
%   paraVec: D x 1 (initial solution)
%   funcHndl: function handler (compute the function's value and gradient)
%   opts: structure (optimization options)
%   varargin: K x 1 (cell array; additional parameters)
% OUTPUT
%   funcVal: scalar (function's value of the optimal solution)
%   funcValLst: T x 1 (list of function's value through iterations)
%   paraVec: D x 1 (optimal solution)

% solve the optimization via gradient-based update
lr = opts.lrInit;
iterIdx = 0;
gradVecAccmFst = zeros(size(paraVec));
gradVecAccmSec = zeros(size(paraVec));
funcValLst = zeros(opts.epchCnt, 1);
for epchIdx = 1 : opts.epchCnt
  % generate the mini-batch partition
  smplIdxLst = GnrtMiniBatc(opts.smplCnt, opts.batcSiz);
  
  % update parameters with mini-batches
  for batcIdx = 1 : numel(smplIdxLst)
    % obtain the function's value and gradient vector of the current solution
    [~, gradVec] = funcHndl(paraVec, smplIdxLst{batcIdx}, varargin{:});

    % compute the adjusted gradient vector
    iterIdx = iterIdx + 1;
    gradVecAccmFst = ...
      opts.betaFst * gradVecAccmFst + (1 - opts.betaFst) * gradVec;
    gradVecAccmSec = ...
      opts.betaSec * gradVecAccmSec + (1 - opts.betaSec) * gradVec .^ 2;
    gradVecAdjsFst = gradVecAccmFst / (1 - opts.betaFst ^ iterIdx);
    gradVecAdjsSec = gradVecAccmSec / (1 - opts.betaSec ^ iterIdx);
    gradVecAdjs = gradVecAdjsFst ./ sqrt(gradVecAdjsSec + opts.fudgFctr);

    % use gradient to update the solution
    paraVec = paraVec - lr * gradVecAdjs;
  end

  % record related variables
  [funcVal, ~] = funcHndl(paraVec, [], varargin{:});
  funcValLst(epchIdx) = funcVal;
end

end
