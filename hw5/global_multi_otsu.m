clear
clc

%读取原图和部分噪声图
ori_img10 = double(rgb2gray(imread('brain.png')));
ori_img20 = double(rgb2gray(imread('heart.png')));
ori_img11 = double(rgb2gray(imread('brain+exp.png')));
ori_img21 = double(rgb2gray(imread('heart+exp.png')));
ori_img12 = double(rgb2gray(imread('brain+rayleigh.png')));
ori_img22 = double(rgb2gray(imread('heart+rayleigh.png')));
ori_img13 = double(rgb2gray(imread('brain+gaussian.png')));
ori_img23 = double(rgb2gray(imread('heart+gaussian.png')));
ori_img14 = double(rgb2gray(imread('brain+salt&pepper.png')));
ori_img24 = double(rgb2gray(imread('heart+salt&pepper.png')));

%用otsu算法进行处理
new_img10 = otsu(ori_img10);
new_img20 = otsu(ori_img20);
new_img11 = otsu(ori_img11);
new_img21 = otsu(ori_img21);
new_img12 = otsu(ori_img12);
new_img22 = otsu(ori_img22);
new_img13 = otsu(ori_img13);
new_img23 = otsu(ori_img23);
new_img14 = otsu(ori_img14);
new_img24 = otsu(ori_img24);

%保存处理结果
imwrite(new_img10, 'brain.png');
imwrite(new_img20, 'heart.png');
imwrite(new_img11, 'brain+exp.png');
imwrite(new_img21, 'heart+exp.png');
imwrite(new_img12, 'brain+rayleigh.png');
imwrite(new_img22, 'heart+rayleigh.png');
imwrite(new_img13, 'brain+gaussian.png');
imwrite(new_img23, 'heart+gaussian.png');
imwrite(new_img14, 'brain+salt&pepper.png');
imwrite(new_img24, 'heart+salt&pepper.png');
function new_img = otsu(ori_img)
[r, c] = size(ori_img);
%每个灰度值的直方图概率初始化
P = zeros(256, 1);
%累积概率初始化
sum_P = P;
%类中平均灰度值初始化
m = P;
%类间方差初始化
var = P;
for i = 1 : r
    for j = 1 : c
        P(ori_img(i, j)+1) = P(ori_img(i, j)+1) + 1;
    end
end
P = P / (r*c);
sum_P(1) = P(1);
for i = 2 : 256
    sum_P(i) = sum_P(i-1) + P(i);
    m(i) = m(i-1) + (i-1) * P(i);
end
mean = m(256);
for i = 1 : 256
    var(i) = (mean * sum_P(i) - m(i))^2 / (sum_P(i) * (1-sum_P(i)));
end
% 按照最大的三个类间方差将图像灰度分为四类
[~, index] = sort(var);
index = sort(index(254:256));
t1 = ori_img <= index(1) - 1;
t2 = (ori_img > index(1)-1) & (ori_img <= index(2)-1);
t3 = (ori_img > index(2)-1) & (ori_img <= index(3)-1);
t4 = ori_img > index(3) - 1;
new_img = colouring(t1, t2, t3, t4);
end

function new_img = colouring(t1, t2, t3, t4)
[r, c] = size(t1);
R1 = zeros(r, c);
G1 = R1;
B1 = R1;

% 用红绿蓝黄四种颜色上色
R1(t1) = 255;
G1(t1) = 0;
B1(t1) = 0;

R1(t2) = 0;
G1(t2) = 255;
B1(t2) = 0;

R1(t3) = 0;
G1(t3) = 0;
B1(t3) = 255;

R1(t4) = 255;
G1(t4) = 255;
B1(t4) = 0;

new_img(:, :, 1) = R1;
new_img(:, :, 2) = G1;
new_img(:, :, 3) = B1;
end