clear
clc

%读取原图和部分噪声图
ori = 'Ori\';
ori_img10 = double(imread([ori,'brain.png']));
[r,c] = size(ori_img10)
ori_img20 = double(imread([ori,'heart.png']));
ori_img13 = double(imread([ori,'brain+salt&pepper.png']));
ori_img23 = double(imread([ori,'heart+salt&pepper.png']));


k = 2;
%用kmeans算法进行k类分割处理
new_img10 = Kmeans(ori_img10, k);
new_img20 = Kmeans(ori_img20, k);
new_img13 = Kmeans(ori_img13, k);
new_img23 = Kmeans(ori_img23, k);


%保存处理结果
T = 'T2\';
imwrite(new_img10,[T,'K brain.png']);
imwrite(new_img20, [T,'K heart.png']);
imwrite(new_img13, [T,'K brain+salt&pepper.png']);
imwrite(new_img23, [T,'K heart+salt&pepper.png']);




function new_img = colouring(idx, r, c, k)
new_img_info = reshape(idx, r, c);
new_img = zeros(r, c, 3);
R1 = zeros(r, c);
G1 = R1;
B1 = R1;
switch k
    case 2
        % 用黑白两种颜色上色
        R1(new_img_info==1) = 0;
        G1(new_img_info==1) = 0;
        B1(new_img_info==1) = 0;
        R1(new_img_info==2) = 255;
        G1(new_img_info==2) = 255;
        B1(new_img_info==2) = 255;
    case 3
        % 用红绿蓝三种颜色上色
        R1(new_img_info==1) = 255;
        G1(new_img_info==1) = 0;
        B1(new_img_info==1) = 0;
        
        R1(new_img_info==2) = 0;
        G1(new_img_info==2) = 255;
        B1(new_img_info==2) = 0;
        
        R1(new_img_info==3) = 0;
        G1(new_img_info==3) = 0;
        B1(new_img_info==3) = 255;
    case 4
        % 用红绿蓝黄四种颜色上色
        R1(new_img_info==1) = 255;
        G1(new_img_info==1) = 0;
        B1(new_img_info==1) = 0;
        
        R1(new_img_info==2) = 0;
        G1(new_img_info==2) = 255;
        B1(new_img_info==2) = 0;
        
        R1(new_img_info==3) = 0;
        G1(new_img_info==3) = 0;
        B1(new_img_info==3) = 255;
        
        R1(new_img_info==4) = 255;
        G1(new_img_info==4) = 255;
        B1(new_img_info==4) = 0;
    otherwise
        error('please input k between 2 and 4') 
end

new_img(:, :, 1) = R1;
new_img(:, :, 2) = G1;
new_img(:, :, 3) = B1;

end

function new_img = Kmeans(ori_img, k)
[r, c, ~] = size(ori_img);
mini = min(ori_img, [], 'all');
maxi = max(ori_img, [], 'all');
% 利用随机数初始化簇中心
centor0 = rand(k, 3) * 255;
centor1 = rand(k, 1) * (maxi-mini) + mini;

% 收敛阈值
err = 1e-12;
% 将RGB分量各转为kmeans数据格式
R = reshape(ori_img(:, :, 1), r*c, 1);    
G = reshape(ori_img(:, :, 2), r*c, 1);
B = reshape(ori_img(:, :, 3), r*c, 1);
data = [R G B];
% idx记录每个点的聚类中心
idx = zeros(r*c, 1);

% 结果不收敛时继续循环
while(norm(centor1 - centor0, 'fro') >= err)
    centor0 = centor1;
    count = zeros(k, 1); % 每类点个数
    
    for i = 1 : r*c
        % 找到每个点对应的距离最近的聚类中心index
        [~, index] = min(sum((centor0 - data(i)).^2, 2));
        idx(i) = index;
        count(index) = count(index) + 1;
    end
    
    % 更新聚类中心
    for i = 1 : k
        centor1(i) = sum(data(idx==i), 1) / count(i);
    end
end

% 为分割的图像上色
new_img = colouring(idx, r, c, k);


end