function [funcVal, paraVec] = minFunc(paraVec, funcHndl, opts, varargin)
% INTRO
%   minimize the objective function via gradient-based methods
% INPUT
%   paraVec: D x 1 (initial solution)
%   funcHndl: function handler (compute the function's value and gradient)
%   opts: structure (optimization options)
%   varargin: K x 1 (cell array; additional parameters)
% OUTPUT
%   funcVal: scalar (function's value of the optimal solution)
%   paraVec: D x 1 (optimal solution)

% choose the proper entry based on the selected optimization method
switch opts.method
  case 'GradDst'
    [funcVal, funcValLst, paraVec] = ...
      minFunc_GradDst(paraVec, funcHndl, opts, varargin{:});
  case 'AdaGrad'
    [funcVal, funcValLst, paraVec] = ...
      minFunc_AdaGrad(paraVec, funcHndl, opts, varargin{:});
  case 'AdaDelta'
    [funcVal, funcValLst, paraVec] = ...
      minFunc_AdaDelta(paraVec, funcHndl, opts, varargin{:});
  case 'Adam'
    [funcVal, funcValLst, paraVec] = ...
      minFunc_Adam(paraVec, funcHndl, opts, varargin{:});
end

% display the objective function's value curve
if opts.enblVis
  figure;
  plot(1 : numel(funcValLst), funcValLst);
  title(opts.method);
  grid on;
  drawnow;
end

end
