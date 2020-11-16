function svm = SOSVEN(x, Y, p)
% function svm = SOSVEN(x, y, p)
% this code optimizes values of t and lambda using 2-way interpolation method
% and one against all for multiple classes

% input Y is a binary encoded matrix of classes or a real encoded matrix of
% properties
% Y also can be input as a cell array with Y{1} being the NsxNk values for
% Ns samples and Nk properties.  Y{2} is a binary array of M measurements
% and N replicates.  Using this mode, replicates will not span the
% model-building and prediction sets of the internal bootstrap Latin
% partitions.
% inputs t and lambda are the values converted from log10(t) and log10(lambda).
% p is an object that defines the parameters and its defaults are as follow,
% defaults
%     p.minlogt = -2;  log10(t)
%     p.maxlogt = 2;
%     p.minlogl = -2;  log10(lambda)
%     p.maxlogl = 2;
%     p.Adesign = 10;
%     p.plotflag = false;
%     p.nboot = 10;
%     p.npart = 2;
%     p.tinterp = 100;
%     p.Lainterp = 100;

% Version 1.0 16-Nov-2020
%
% author: Peter.Harrington@OHIO.edu
% it uses the first principal component scores to sort the rows of the y matrix
% if you use this routine then please reference the following paper
% Version 2 29-7Jul-07
% partitions by object same y values may be split between partitions
% good for splitting samples
%
% Please cite the below if you use this method
% https://dx.doi.org/10.1021/acs.analchem.0c01506.
%
% author: Peter.Harrington@OHIO.edu
%
% Harrington, P.B. Statistical validation of classification and calibration
% models using bootstrapped Latin partitions.
% Trac-Trends in Analytical Chemistry 2006, 25, 1112-1124.
%
% P.B. Harrington, Multiple Versus Single Set Validation to Avoid Mistakes,
% CRC Critical Reviews in Analytical Chemistry.
% (2017) 48(1) 1-14 DOI: 10.1080/10408347.2017.1361314.
% You may distribute this code freely
%
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from the author.
%
% The programs and documents are distributed without any warranty, expressed or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.
%
%***************************************************************************
% default parameters
if nargin < 3
    p = [];
end
if ~isfield(p, 'minlogt')
    p.minlogt = -1;
end
if ~isfield(p, 'maxlogt')
    p.maxlogt = 1;
end
if ~isfield(p, 'minlogl')
    p.minlogl = -3;
end
if ~isfield(p, 'maxlogl')
    p.maxlogl = 0;
end
if ~isfield(p, 'ldesign')
    p.ldesign = 10;
end
if ~isfield(p, 'tdesign')
    p.tdesign = 10;
end
if ~isfield(p, 'PlotFlag')
    p.PlotFlag = true;
end
if ~isfield(p, 'nboot')
    p.nboot = 10;
end
if ~isfield(p, 'npart')
    p.npart = 2;
end
if ~isfield(p, 'tinterp')
    p.tinterp = 100;
end
if ~isfield(p, 'Lainterp')
    p.Lainterp = 100;
end

%% generate design points
lx1 = linspace(p.minlogt, p.maxlogt, p.tdesign);
lx2 = linspace(p.maxlogl, p.minlogl, p.ldesign);
[meshx1, meshx2] = meshgrid(lx1, lx2);

x1 = 10.^lx1;
x2 = 10.^lx2;
mT = length(x1);
mL = length(x2);

%% big matrix to find the optimal
lx11 = linspace(p.minlogt, p.maxlogt, p.tinterp);% more points in log10(t)
lx22 = linspace(p.maxlogl, p.minlogl, p.Lainterp);% more points for log10(lambda)


[m, n] = size(x);

lseq = false(p.nboot, m, p.npart);
if iscell(Y)
    ind = sum(Y{2}) > 0;
    Y{2} = Y{2}(:, ind);
    for i = 1:p.nboot
        lseq(i, :, :) = Latin(Y, p.npart);
    end
    yall = Y{2}*Y{1};
else
    for i = 1:p.nboot
        lseq(i, :, :) = Latin(Y, p.npart);
    end
    yall = Y;
end
[my, ny] = size(yall);

if isempty(setdiff(yall, [0, 1]))
    FLAG = true;
else
    FLAG = false;
end

if m ~= my
    fprintf(1, 'Error: number of rows of x and y do not correspond');
    return
end

Hits = zeros(p.nboot, mT, mL);
%%
trainx = x(~lseq(1, :, 1), :);
trainy = yall(~lseq(1, :, 1), 1);
f = @() SVENP(trainx, trainy,...
    median(x1), median(x2));
tP = timeit(f);
f = @() SVEND(trainx, trainy,...
    median(x1), median(x2));
tD = timeit(f);
if tD <= tP
    DualF = true;
else
    DualF = false;
end

for iii = 1:ny
    biga1 = zeros(mT, mL, 2*n);
    icnt = 0;
    % split training set and prediction set
    for ii = 1:p.nboot
        for k = 1:p.npart
            icnt = icnt + 1;
            predx = x(lseq(ii, :, k), :);
            trainx = x(~lseq(ii, :, k), :);
            trainy = yall(~lseq(ii, :, k), iii);
            i1 = squeeze(yall(lseq(ii, :, k), iii));
            % internal evaluation
            for i = 1:mT
                for j = 1:mL
                    if DualF
                        s = SVEND(trainx, trainy,...
                            x1(i), x2(j), biga1(i, j, :));
                    else
                        s = SVENP(trainx, trainy,...
                            x1(i), x2(j), biga1(i, j, :));
                    end
                    biga1(i, j, :) = (s.a0 + (icnt - 1)*squeeze(biga1(i, j, :)))/icnt;
                    p1 = predx*s.w + s.b;
                    if FLAG
                        p1 = p1 > 0.5;
                    end
                    Hits(ii, i, j) = Hits(ii, i, j) + sum((p1 - i1).^2);
                end
            end
        end
    end
    
    mAve = squeeze(mean(Hits, 1)); % average for same element from nboot slices
    mAve = sqrt(mAve/m);
    
    % 2-way interpolation
    ymodel = interp2(lx2', lx1, mAve, lx22', lx11, 'spline');
    ind = find(min(ymodel(:))==ymodel);
    [i, j] = ind2sub(size(ymodel), ind);
    
    %% the optimal model
    t = 10.^lx11(i);
    lambda = 10.^lx22(j);
    minvar = inf;
    a1 = [];
    for ii = 1:length(t)
        if DualF
            s = SVEND(x, yall(:, iii), t(ii), lambda(ii), a1);
        else
            s = SVENP(x, yall(:, iii), t(ii), lambda(ii), a1);
        end
        svm1.w(:, iii) = s.w;
        svm1.b(iii) = s.b;
        svm1.v(:, iii) = abs(s.w) > 0;
    end
    vopt = 100*sum(max(svm1.v, [], 2))/n;
    if vopt < minvar
        %         minvar = vopt;
        svm = svm1;
        iopt = ii;
    end
    
    svm.ny = ny;
    %% surface plot
    if p.PlotFlag
        figure
        box on;
        imagesc(lx22, lx11, ymodel);
        set(gca, 'YDir', 'normal');
        h = colorbar;
        ylabel(h, 'Estimated Error (%)', 'Fontsize', 16)
        hold on
        % plot design points
        plot(meshx2(:), meshx1(:),'k+', 'Markerface', 'k', 'Markersize', 8);
        %plot the optimal point
        plot(lx22(j(iopt)), lx11(i(iopt)), 'w^', 'Markerface', 'w', 'Markersize', 8);
        hold on
        text(lx22(j(iopt)),lx11(i(iopt)), sprintf('%3.f%% Size', round(vopt)));
        
        set(gca, 'YDir', 'normal');
        ylabel('Log_1_0(\itt\rm) ',  'Fontsize', 16);
        xlabel('Log_1_0(\lambda)',  'Fontsize', 16);
        hold off
       
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = SVENP(x, y, t, lambda, a0)
% s = SVENP(x, y, t, lambda)
% Primal SVEN
% support vector elastic net quadratic loss
% general svm code for primal and dual calculations
% x should be mean centered.
ZEPS = 1e-5;

if nargin < 5
    a0 = [];
end

[m1, n1] = size(x);

options = optimoptions('quadprog', 'Display', 'none');
mx = mean(x);
x1 = x - mx;

my = mean(y);
y1 = y - my;
x2 = [x1-y1/t, x1+y1/t]';
y2 = [ones(n1, 1); -ones(n1, 1)];
C = 1/(2*lambda*m1);

[m, n] = size(x2);

Z = y2.*x2;
% primal Gaussian
lb = [-inf(1, n), zeros(1, m)]';
H = ones(m + n, 1);
H(n+1:m+n) = C;
%     H = diag(H);
H = spdiags(H, 0, m+n, m+n);
%     Aeq = [ones(1, n), zeros(1, m)];
%     beq = 1;
[a0, fval, exitflag] = quadprog(H, [], -[Z, eye(m)], -ones(m, 1), [],...
    [], lb, [], a0, options);

w = a0(1:n, 1);
s.a0 = C * max(1-y2.*(x2*w),0);
s.w = t * (s.a0(1:n1) - s.a0(n1+1:2*n1)) / sum(s.a0);

s.w(abs(s.w) < ZEPS) = 0;
yhat = x*s.w;
s.b = my - mean(yhat);
% s.b = a0'*(y-x*s.w)/sum(a0);
s.ef  = exitflag;
s.ny = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = SVEND(x, y, t, lambda, a0)
% s = SVEND(x, y, t, lambda, a0)
% Dual SVEN
% support vector elastic net quadratic loss
% general svm code for primal and dual calculations
% x should be mean centered.

ZEPS = 1e-5;

if nargin < 5
    a0 = [];
end

[m1, n1] = size(x);

options = optimoptions('quadprog', 'Display', 'none');
mx = mean(x);
x1 = x - mx;

my = mean(y);
y1 = y - my;
x2 = [x1-y1/t, x1+y1/t]';
y2 = [ones(n1, 1); -ones(n1, 1)];
C = 1/(2*lambda*m1);

[m, n] = size(x2);

Z = y2.*x2;
% dual Gaussian
K = Z*Z' + eye(m)/C;
[s.a0, fval, exitflag, output] = quadprog(K, -ones(m, 1), [], [], [], [],...
    zeros(m, 1), inf(m, 1), a0, options);
s.w = t * (s.a0(1:n1) - s.a0(n1+1:2*n1)) / sum(s.a0);

s.w(abs(s.w) < ZEPS) = 0;
yhat = x*s.w;
s.b = my - mean(yhat);

s.ef  = exitflag;
s.ny = 1;
end
