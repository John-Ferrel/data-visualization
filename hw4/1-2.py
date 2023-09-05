import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

def white_noise(img):
    f = np.fft.fft2(img)
    m, n = img.shape
    x,y = math.floor(m/2)-20, math.floor(n/2)-20
    fshift = np.fft.fftshift(f)
    fshift[x, y] = 11115984.30350322+5.04331046e+03j
    fshift[m - x, n - y] = 11115984.30350322+5.04331046e+03j
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    return new_img


def gauss_noise(img, mean=0, var=0.1):
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    noisy = img / 256 + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy




# read image
img = cv.imread('test2.tif', 0)

# show the origin image
print(type(img),img.dtype, img.shape)
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()

# process
noisy1 = gauss_noise(img, 0, 0.1)
noisy2 = white_noise(img)
# show the result
plt.imshow(noisy1,cmap="gray")
plt.axis('off')
plt.show()
plt.imshow(noisy2,cmap="gray")
plt.axis('off')
plt.show()