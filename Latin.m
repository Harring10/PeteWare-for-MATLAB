function lseq = Latin(Y, npart)
% lseq = Latin(Y, npart)
% general partition algorithm for classification and calibration
% split object us a single Y matrix
% split samples so that replicates are never split use cell array
% Y{1} contains sample values
% y{2} is a binary matrix of nspectra x nsamp that defines the replicates
% returns logical values for test sets with each column in a partition
% empty sequences are coded as zeros
% assumes class i.d.'s are encoded as 1s and 0s in Y for classifiers
% returns npar mutual exclusive partitions with a uniform span in y
% uses the first principal component to sort the rows of the y matrix
% if you use this routine then please reference the following papers

% Version 3 10-Jun-2020
% partitions by sample
% Harrington, P.B. Statistical validation of classification and calibration
% models using bootstrapped Latin partitions.
% Trac-Trends in Analytical Chemistry 2006, 25, 1112-1124.
%
% author: Peter.Harrington@OHIO.edu
% it uses the first principal component scores to sort the rows of the y matrix
% if you use this routine then please reference the following paper
% Version 2 29-7Jul-07
% partitions by object same y values may be split between partitions
% good for splitting samples
%
% Please cite the below if you use this method
%
%
% author: Peter.Harrington@OHIO.edu
%
% C. Wan and P.B. Harrington, Screening GC-MS data for carbamate pesticides
% with temperature-constrained-cascade correlation neural networks.
% Analytica Chimica Acta 2000, 408, 1-12.
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
if iscell(Y)
    ind = sum(Y{2}) > 0;
    Y{2} = Y{2}(:, ind);
    if isempty(setdiff(Y{1}, [0, 1]))
        lseq1 = LatinPart(Y{1}, npart);
        lseq = Y{2}*lseq1;
    else
        lseq1 = LatinCal(Y{1}, npart);
        lseq = Y{2}*lseq1;
    end
else
    if isempty(setdiff(Y, [0, 1]))
        lseq = LatinPart(Y, npart);
    else
        lseq = LatinCal(Y, npart);
    end
end
end
%*************************************************************************
function lseq = LatinPart(y, npart)
[my, ny] = size(y);
nlen = ceil(my/npart);

tpart = [];
if ny > 1
    for j=1:ny
        ind = find(y(:, j) > 0);
        mind = length(ind);
        tpart = [tpart; ind(randperm(mind))];
    end
else
    ind1 = find(y);
    ind2 = find(y==0);
    mind1 = length(ind1);
    mind2 = length(ind2);
    tpart = [ind1(randperm(mind1)); ind2(randperm(mind2))];
end

npad = npart*nlen-my;
tpart = [tpart; zeros(npad, 1)];

tpart = reshape(tpart, npart, nlen)';
% remove padding
for j=1:npart
    i = tpart(:, j) > 0;
    tpart1{j} = tpart(i, j);
end
lseq = zeros(my, npart);
for j = 1:npart
    lseq(tpart1{j}, j) = 1;
end
lseq = logical(lseq);
end
%*************************************************************************
function bigseq = LatinCal(y, npar)
[m, ny] = size(y);
if ny > 1
    if m > ny
        [u, s, v] = svd(y-mean(y), 0);
    else
        [v, s, u] = svd((y-mean(y))', 0);
    end
    [sy, iy] = sortrows(u*s, 1);
else
    [sy, iy] = sort(y);
end
mpar = ceil(m/npar);
madd = mpar*npar-m;

y = [y; inf(madd, ny)];
iy = [iy' (m+1):(m+madd)]';

iseq = reshape(iy, npar, mpar)';

% now shuffle the columns of each row

for i=1:mpar
    j = randperm(npar);
    iseq(i, :) = iseq(i, j);
end

ind = y(iseq) == inf;
iseq(ind) = 0;

% build full seq array
bigseq = false(m, npar);
for jj = 1:npar
    seq = iseq(:, jj);
    seq = seq(seq > 0);
    bigseq(seq, jj) = true;
end
end