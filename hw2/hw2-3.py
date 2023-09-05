import numpy as np
import cv2 as cv
import matplotlib.pylab as plt


def inter_line(im,n=2):
    w,h = im.shape
    w1,h1 = n*w,n*h
    newimg = np.zeros((w1,h1))
    for x1 in range(w1):
        for y1 in range(h1):
            # original image left top point
            x = x1 / n
            y = y1 / n
            x_int = x1 // n
            y_int = y1 // n
            a = x - x_int
            b = y - y_int

            if x_int + 1 == w or y_int + 1 == h:
                newimg[x1, y1] = im[x_int, y_int]
            else:
                newimg[x1, y1] = int((1. - a) * (1. - b) * im[x_int + 1, y_int + 1] + \
                                  (1. - a) * b * im[x_int, y_int + 1] + a * (1. - b) * im[x_int + 1, y_int]\
                               + a * b * im[x_int, y_int])

    return newimg
# read picture
img = cv.imread('img3.jpg',0)
print(img.shape, img.dtype, type(img))

# show image
plt.imshow(img,cmap = 'gray')
plt.axis('off')
plt.show()

# process
newimg = inter_line(img,20)
print(newimg.shape, newimg.dtype, type(newimg))
# show
plt.imshow(newimg,cmap = 'gray')
plt.axis('off')
plt.show()
# save
loc_gim = cv.imwrite('res3.jpg', img=newimg)