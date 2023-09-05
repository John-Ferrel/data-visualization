import numpy as np
import matplotlib.pylab as plt
import cv2 as cv


def low_filter(im, m=3, n=3):
    h,w = im.shape
    nimg = np.zeros((h, w), np.int16)
    kernel = 1/(m*n) * np.ones((m, n))
    a, b = int((m-1)/2), int((n-1)/2)
    bim = cv.copyMakeBorder(im, a, a, b, b, cv.BORDER_REFLECT_101, value=0)
    for i in range(h):
        for j in range(w):
            nimg[i,j] = np.sum(kernel * bim[i:i+m,j:j+n])
    return  nimg


# load image
img = cv.imread('sleeping.webp', 0)
print(img.shape, img.dtype, type(img))

# show image
plt.imshow(img,cmap = 'gray')
plt.axis('off')
plt.show()

# for i in [3,11,21]:
    # process
i = 5
nimg = low_filter(img, i, i)

# show new image
plt.imshow(nimg,cmap = 'gray')
plt.axis('off')
plt.show()

im = cv.imwrite('sleeping_gray.png',nimg)
