clear
clc

%读取原图和部分噪声图
ori = 'Ori\';
ori_img10 = double(imread([ori,'brain.png']));
ori_img20 = double(imread([ori,'heart.png']));
ori_img13 = double(imread([ori,'brain+salt&pepper.png']));
ori_img23 = double(imread([ori,'heart+salt&pepper.png']));

%用otsu算法进行处理
new_img10 = otsu(ori_img10);
new_img20 = otsu(ori_img20);
new_img11 = otsu(ori_img13);
new_img21 = otsu(ori_img23);

%保存处理结果
ot = 'Otsu\';
imwrite(new_img10, [ot,'brain.png']);
imwrite(new_img20, [ot,'heart.png']);
imwrite(new_img11,[ot, 'brain+salt&pepper.png']);
imwrite(new_img21, [ot,'heart+salt&pepper.png']);
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
[~, thre] = max(var);
threshold = ori_img > thre - 1;
new_img = zeros(r, c);
new_img(threshold) = 1;
new_img = new_img(1:r,1:c/3);
end