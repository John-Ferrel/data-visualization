clc
clear
A = rgb2gray(imread('brain.png'));
B = rgb2gray(imread('heart.png'));
figure
imshow(A)
title('origin image1');
figure
imshow(B)
title('origin image2');
A = double(A);
B = double(B);
T1 = local_(A);
T2 = local_(B);
A = sign(A-T1);
B = sign(B-T2);
figure
imshow(A);
title('after local otsu image1');
figure
imshow(B)
title('after local otsu image2');
% imwrite(A, 'local_otsu1.jpg');
% imwrite(B, 'local_otsu2.jpg');
%%
%-----------------------------------------
%functions used
function T = local_(A)
[M, N] = size(A);
n = 50;
his1 = hist_(A, M, N);
figure
bar(0 : 255, his1);
title('histogram');
%此处默认边界外像素值为0
A = blkdiag(zeros(n, n), A, zeros(n, n));
m = 0;
P1 = ones(256, 1);
m1 = P1;
T = zeros(M, N);
num = n;
for k = n+1 : N+n
    [T(1, k-n), P1, m1, m] = otsu(num, P1, m1, m, A(n+1, k), A(n+1, k-n));
end
%对A作镜像对称，方便局部Z形滑动
A = fliplr(A);
for i = n+2 : M + n
    for j = n+1 : N+n
        x = 2 * n - j;
        if(x >= 0)
            [T(i-n, j-n), P1, m1, m] = otsu(num, P1, m1, m, A(i, j), A(i-1, n+x+1));
        else
            [T(i-n, j-n), P1, m1, m] = otsu(num, P1, m1, m, A(i, j), A(i, j-n));
        end
    end
    A = fliplr(A);  %每次走完一行需要左右翻转一次矩阵
end
end


function [t, P1, m1, m] = otsu(N, P1, m1, m, newin, oldout)
m1 = m1 .* P1;
if newin < oldout
    P1 = P1 + [zeros(newin, 1); 1/N * ones(oldout-newin, 1); zeros(256-oldout, 1)];
elseif newin > oldout
    P1 = P1 - [zeros(oldout, 1); 1/N * ones(newin-oldout, 1); zeros(256-newin, 1)];
end
m1 = m1 + [zeros(newin, 1); newin/N * ones(256-newin, 1)];
m1 = m1 - [zeros(oldout, 1); oldout/N * ones(256-oldout, 1)];
m1 = m1 ./ P1;
m = m + newin/N - oldout/N;
varB = (m.*P1 - m).^2 ./ (P1.*(1-P1));
[~, t] = max(varB);
end

function his = hist_(A_, m, n)
his = zeros(256, 1);
for i = 1 : m
    for j = 1 : n
        his(A_(i, j) + 1, 1) = his(A_(i, j) + 1, 1) + 1;
    end
end
his = his / (m*n);
end