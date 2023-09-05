import numpy as np
import matplotlib.pylab as plt
import cv2 as cv

img = cv.imread('sleeping.webp')
nimg = cv.medianBlur(img,5)
cv.imwrite('sleeping.png',nimg)