clear
clc

ori = 'Ori\';
ori_img = imread([ori,'zmic_fdu_noise.bmp']);

SE = zeros(5,5);

new_img = dila(ori_img,SE);
new_img = corr(new_img,SE);
imshow(new_img)

SE = zeros(5,5);
new_img = corr(new_img,SE);
new_img = dila(new_img,SE);
imshow(new_img)
imwrite(new_img, 'Seg\res.bmp');

function [new_img] = dila(ori_img, SE)
[r1,c1] = size(SE);
[r,c] = size(ori_img);
new_img = ones(r+r1-1,c+c1-1);
for i = 1 : r
    for j = 1 : c 
        if ori_img(i,j) == 0
            new_img(i:i+r1-1,j:j+c1-1) = SE;
        end
    end
end
new_img = new_img(1+(r1-1)/2:r+1,1:c+(c1-1)/2);
end

function [new_img] = corr(ori_img,SE)
SE = 1-SE;
[r1,c1] = size(SE);
[r,c] = size(ori_img);
new_img = zeros(r+r1-1,c+c1-1);
for i = 1 : r
    for j = 1 : c 
        if ori_img(i,j) == 1
            new_img(i:i+r1-1,j:j+c1-1) = SE;
        end
    end
end
new_img = new_img(1+(r1-1)/2:r+1,1:c+(c1-1)/2);
end
