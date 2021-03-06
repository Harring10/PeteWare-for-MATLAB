function t = Scientific(x, prec)
% t = Scientific(x, [prec])
%   Converts a floating poinr number to a text string with scientific
%   notation.
% Peter.Harrington@OHIO.edu 5-April-2020

if nargin == 1
    prec = 1;
end

e = log10(abs(x));
e = floor(e);

p = 10^(prec-1);
x = x/10^e;
mt = round(x*p)/p;
t = sprintf('%g\\times10^{%i}', mt, e);
end

