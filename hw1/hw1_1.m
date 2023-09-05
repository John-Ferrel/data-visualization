% set arguments
k = 0.5;
r = 80;
x = 0 : 1 : 255;
y = pixel_de(k, r, x);

% read the picture
A = double(imread('test1-1.tif'));

% show the picture
figure(1)
imshow(uint8(A))

% process
B = pixel_de(k, r, A);
figure(2)
imshow(uint8(B))
title(['r = ', num2str(r), ', k = ', num2str(k)]);
imwrite(uint8(B), 'result1-1.tif')

% Piecewise function
function m = pixel_de(k, r, t)
m = k.*t.*(t>=0 & t<r) + ((255-2*r.*k)/(255-2*r).*(t-r)+r*k).*(t>=r & t<(255-r)) + (k*(t-255)+255).*(t>=(255-r) & t<=255);
end