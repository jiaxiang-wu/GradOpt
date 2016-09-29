function demo()
% INTRO
%   demonstrates the usage of minFunc(), in the context of linear regression
% INPUT
%   None
% OUTPUT
%   None

close all; clearvars; clc;

% randomly generate feature vectors and targe values
opts.smplCnt = 1000;
opts.featCnt = 100;
opts.ouptCnt = 50;
opts.featMgn = 1.0;
opts.projMgn = 1.0;
opts.noisRat = 0.01;
[featMat, ouptMat, projMatUnly] = GnrtData(opts);

% evaluate the underlying projection matrix
[funcValUnly, ~] = CalcFuncGrad(projMatUnly(:), [], featMat, ouptMat);
fprintf('[INFO] funcVal (underlying) = %.4e\n', funcValUnly);

% obtain the closed-form solution to the linear regression
projMatClsd = (ouptMat / featMat)';
[funcValClsd, ~] = CalcFuncGrad(projMatClsd(:), [], featMat, ouptMat);
fprintf('[INFO] funcVal (closed-form) = %.4e\n', funcValClsd);

% evaluate the optimization method
EvaMethod_GradDst(featMat, ouptMat);
EvaMethod_AdaGrad(featMat, ouptMat);
EvaMethod_AdaDelta(featMat, ouptMat);
EvaMethod_Adam(featMat, ouptMat);

end

function [featMat, ouptMat, projMat] = GnrtData(opts)
% INTRO
%   randomly generate a dataset for linear regression
% INPUT
%   opts: structure (dataset generation options)
% OUTPUT
%   featMat: D x N (feature matrix)
%   ouptMat: R x N (output matrix)
%   projMat: D x R (projection matrix)

% randomly generate feature vectors and targe values
featMat = opts.featMgn * randn(opts.featCnt, opts.smplCnt);
projMat = opts.projMgn * randn(opts.featCnt, opts.ouptCnt);
ouptMat = projMat' * featMat;
noisMgn = opts.noisRat * norm(ouptMat, 'fro');
ouptMat = ouptMat + noisMgn * randn(opts.ouptCnt, opts.smplCnt);

end

function [funcVal, gradVec] = CalcFuncGrad(paraVec, smplIdxs, featMat, ouptMat)
% INTRO
%   compute the objective function's value and gradient vector
% INPUT
%   paraVec: (D * R) x 1 (projection matrix, viewed as the column vector)
%   smplIdxs: M x 1 (list of sample indexes)
%   featMat: D x N (feature matrix)
%   ouptMat: R x N (output matrix)
% OUTPUT
%   funcVal: scalar (objective function's value)
%   gradVec: (D * R) x 1 (objective function's gradient vector)

% obtain basic variables
featCnt = size(featMat, 1);
smplCnt = size(featMat, 2);
ouptCnt = size(ouptMat, 1);

% recover the projection matrix
projMat = reshape(paraVec, [featCnt, ouptCnt]);

% construct a mini-batch with randomly selected samples
if ~isempty(smplIdxs)
  batcSiz = numel(smplIdxs);
  featMat = featMat(:, smplIdxs);
  ouptMat = ouptMat(:, smplIdxs);
else
  batcSiz = smplCnt;
end

% compute the objective function's value
diffMat = projMat' * featMat - ouptMat;
funcVal = norm(diffMat, 'fro') ^ 2 / 2 / batcSiz;

% compute the objective function's gradient vector
gradMat = featMat * diffMat' / batcSiz;
gradVec = gradMat(:);

end

function projMatIter = EvaMethod_GradDst(featMat, ouptMat)
% INTRO
%   evaluate the optimization method: gradient descent
% INPUT
%   featMat: D x N (feature matrix)
%   ouptMat: R x N (output matrix)
% OUTPUT
%   projMatIter: D x R (projection matrix, obtained via iterative optimization)

% obtain basic variables
featCnt = size(featMat, 1);
smplCnt = size(featMat, 2);
ouptCnt = size(ouptMat, 1);

% configure optimization options
opts.method = 'GradDst';
opts.enblVis = true;
opts.epchCnt = 100;
opts.smplCnt = smplCnt;
opts.batcSiz = 50;
opts.lrInit = 1e-2;
opts.momentum = 0.9;

% solve the optimization via gradient descent
projMatInit = randn(featCnt, ouptCnt);
paraVecInit = projMatInit(:);
[funcVal, paraVecIter] = ...
  minFunc(paraVecInit, @CalcFuncGrad, opts, featMat, ouptMat);
projMatIter = reshape(paraVecIter, [featCnt, ouptCnt]);
fprintf('[INFO] funcVal (%s) = %.4e\n', opts.method, funcVal);

end

function projMatIter = EvaMethod_AdaGrad(featMat, ouptMat)
% INTRO
%   evaluate the optimization method: AdaGrad
% INPUT
%   featMat: D x N (feature matrix)
%   ouptMat: R x N (output matrix)
% OUTPUT
%   projMatIter: D x R (projection matrix, obtained via iterative optimization)

% obtain basic variables
featCnt = size(featMat, 1);
smplCnt = size(featMat, 2);
ouptCnt = size(ouptMat, 1);

% configure optimization options
opts.method = 'AdaGrad';
opts.methodDisp = 'AdaGrad';
opts.enblVis = true;
opts.epchCnt = 100;
opts.smplCnt = smplCnt;
opts.batcSiz = 50;
opts.lrInit = 1e-1;
opts.autoCorr = 0.95;
opts.fudgFctr = 1e-6;

% solve the optimization via AdaGrad
projMatInit = randn(featCnt, ouptCnt);
paraVecInit = projMatInit(:);
[funcVal, paraVecIter] = ...
  minFunc(paraVecInit, @CalcFuncGrad, opts, featMat, ouptMat);
projMatIter = reshape(paraVecIter, [featCnt, ouptCnt]);
fprintf('[INFO] funcVal (%s) = %.4e\n', opts.method, funcVal);

end

function projMatIter = EvaMethod_AdaDelta(featMat, ouptMat)
% INTRO
%   evaluate the optimization method: AdaDelta
% INPUT
%   featMat: D x N (feature matrix)
%   ouptMat: R x N (output matrix)
% OUTPUT
%   projMatIter: D x R (projection matrix, obtained via iterative optimization)

% obtain basic variables
featCnt = size(featMat, 1);
smplCnt = size(featMat, 2);
ouptCnt = size(ouptMat, 1);

% configure optimization options
opts.method = 'AdaDelta';
opts.methodDisp = 'AdaDelta';
opts.enblVis = true;
opts.epchCnt = 100;
opts.smplCnt = smplCnt;
opts.batcSiz = 50;
opts.momentum = 0.999;
opts.fudgFctr = 1e-6;

% solve the optimization via AdaDelta
projMatInit = randn(featCnt, ouptCnt);
paraVecInit = projMatInit(:);
[funcVal, paraVecIter] = ...
  minFunc(paraVecInit, @CalcFuncGrad, opts, featMat, ouptMat);
projMatIter = reshape(paraVecIter, [featCnt, ouptCnt]);
fprintf('[INFO] funcVal (%s) = %.4e\n', opts.method, funcVal);

end

function projMatIter = EvaMethod_Adam(featMat, ouptMat)
% INTRO
%   evaluate the optimization method: Adam
% INPUT
%   featMat: D x N (feature matrix)
%   ouptMat: R x N (output matrix)
% OUTPUT
%   projMatIter: D x R (projection matrix, obtained via iterative optimization)

% obtain basic variables
featCnt = size(featMat, 1);
smplCnt = size(featMat, 2);
ouptCnt = size(ouptMat, 1);

% configure optimization options
opts.method = 'Adam';
opts.methodDisp = 'Adam';
opts.enblVis = true;
opts.epchCnt = 100;
opts.smplCnt = smplCnt;
opts.batcSiz = 50;
opts.lrInit = 1e-1;
opts.betaFst = 0.90;
opts.betaSec = 0.90;
opts.fudgFctr = 1e-6;

% solve the optimization via AdaDelta
projMatInit = randn(featCnt, ouptCnt);
paraVecInit = projMatInit(:);
[funcVal, paraVecIter] = ...
  minFunc(paraVecInit, @CalcFuncGrad, opts, featMat, ouptMat);
projMatIter = reshape(paraVecIter, [featCnt, ouptCnt]);
fprintf('[INFO] funcVal (%s) = %.4e\n', opts.method, funcVal);

end
