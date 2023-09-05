import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def PassFilter(image, d,p='low'):
    m, n = image.shape
    image = cv.copyMakeBorder(image, 0, m, 0, n, cv.BORDER_REFLECT_101)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    def cal_distance(pa, pb):
        dis = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
        return dis

    def make_transform_matrix(d,p):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, image.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):

                dis = cal_distance(center_point, (i, j))
                if p == 'low':
                    k = 1
                else:
                    k = 0
                if dis <= d:
                    transfor_matrix[i, j] = k
                else:
                    transfor_matrix[i, j] = 1-k
        return transfor_matrix

    d_matrix = make_transform_matrix(d,p)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return magnitude_spectrum,new_img[:m, :n]


# read image
img = cv.imread('test1.tif',0)

# show the origin image
print(type(img),img.dtype,img.shape)
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()

# process
magnitude_spectrum,low_new_img = PassFilter(img,60,p='low')
_,high_new_img = PassFilter(img,60,p='high')
# show the result
plt.imshow(magnitude_spectrum,cmap="gray")
plt.axis('off')
plt.show()
plt.imshow(low_new_img,cmap="gray")
plt.axis('off')
plt.show()
plt.imshow( high_new_img,cmap="gray")
plt.axis('off')
plt.show()