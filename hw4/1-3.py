import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def notch_filter(img,d=3):
    m, n = img.shape
    img = cv.copyMakeBorder(img, 0, m, 0, n, cv.BORDER_REFLECT_101)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    abs_max = np.max(np.abs(fshift))
    thre_h = abs_max / 1.3
    thre_l = abs_max/2500
    magnitude_spectrum1 = 20 * np.log(np.abs(fshift)+1)
    def make_transform_matrix(d):
        transfor_matrix = np.ones(img.shape)
        m,n = transfor_matrix.shape
        for i in range(d,m -d):
            for j in range(d,n -d):
                if ((i < 22/50 * m or i > 28 /50 * m) and (j < 22/50 * n or j > 28 /50 * n )) \
                        and thre_l < np.abs(fshift[i][j]) < thre_h:
                    transfor_matrix[i-d:i + d,j-d:j+d] = 1/2500
            print(i)
        return transfor_matrix

    d_matrix = make_transform_matrix(d)
    magnitude_spectrum2 = 20 * np.log(np.abs(fshift * d_matrix)+1)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img[:m, :n],magnitude_spectrum1,magnitude_spectrum2


# read image
img = cv.imread('test3.PNG', 0)

# show the origin image
print(type(img),img.dtype, img.shape)
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()

# process
new_img,p1,p2 = notch_filter(img)

# show the result
plt.imshow(new_img,cmap="gray")
plt.axis('off')
plt.show()
plt.imshow(p1,cmap="gray")
plt.axis('off')
plt.show()
plt.imshow(p2,cmap="gray")
plt.axis('off')
plt.show()