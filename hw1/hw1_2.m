% clean the buffer
clc
clear

% read the picture
A = imread('hw1-2.jpg');

% process
h = joint_histogram(A, 2);

% 2D histogram process function
function h=joint_histogram(Z, K)
r = size(Z, 1);
c = size(Z, 2);
n = size(Z, 3);
if n < K
    disp("Error!not enough dimensions")
    quit
end

% initialize a K-dimensional matrix
N = 256;
h = zeros(ones(1, K) * N);   

% update the date
for i = 1 : r
    for j = 1 : c
        a = zeros(K, 1);
        for k = 1 : K
            a(k, 1) = Z(i, j, k) + 1;
        end
        a1 = num2cell(a);
        h(a1{:}) = h(a1{:}) + 1;
    end
end
figure
hb = bar3(h);
title('joint-histogram')
shading interp
for k = 1 : length(hb)
    zdata = hb(k).ZData;
    hb(k).CData = zdata;
    hb(k).FaceColor = 'interp';
end
colorbar
figure
imagesc(log(h))
title('log of sum')
colorbar
end
