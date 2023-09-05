import numpy as np
import matplotlib.pylab as plt
import cv2 as cv

def high_filter(im,kernel):
    h,w = im.shape
    nimg = np.zeros((h, w), np.int16)
    bim = cv.copyMakeBorder(im, 1, 1, 1, 1, cv.BORDER_REPLICATE, value=0)
    k = 3
    for i in range(h):
        for j in range(w):
            nimg[i,j] = im[i,j] - np.sum(kernel * bim[i:i+k,j:j+k])
    return nimg

# load image
img = cv.imread('test1-2.tif', 0)
print(img.shape, img.dtype, type(img))

# show origin image
plt.imshow(img,cmap = 'gray')
plt.axis('off')
plt.show()

# process
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
nimg = high_filter(img,kernel)

# show new image
plt.imshow(nimg, cmap='gray')
plt.axis('off')
plt.show()