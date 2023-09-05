% clean the buffer
clc
clear
tic

% read the picture
A = imread('test1-3.tif');
subplot(211)

% show the origin picture
imshow(A)
title("origin picture")

% global histogram equalization
A = double(A);
[mean, var] = mv(A);

% set the arguments
n = 3;
[r, c] = size(A);

% update the local hist
for i = (n/2+1) : (r-n/2)
    for j = (n/2+1) : (c-n/2)
        M = A(i-n/2 : i+n/2, j-n/2 : j+n/2);
        M = hist_eq(M, 4, 0.4, 0.02, 0.4, mean,  var);
        A(i-n/2 : i+n/2, j-n/2 : j+n/2) = M;
    end
end
A = uint8(A);
subplot(212)
imshow(A)
imwrite(A, 'result1-3.jpg')
title("after local histogram equalization")
toc

function [mean, var] = mv(A)
[r, c] = size(A);
mean = 0;
var2 = 0;
for i = 1 : r
    for j = 1 : c
        mean = mean + A(i, j);
    end
end
mean = mean / numel(A);
for i = 1 : r
    for j = 1 : c
        var2 = var2 + (A(i, j) - mean)^2;
    end
end
var = sqrt(var2);
end

function A = hist_eq(A, E, k0, k1, k2, mean,  var)
[mean1, var1] = mv(A);
if mean1 <= k0 * mean && k1 * var <= var1 <= k2 * var
    A = E .* A;
end
end