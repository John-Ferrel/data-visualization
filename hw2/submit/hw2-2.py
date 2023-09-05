import numpy as np
import cv2 as cv
import matplotlib.pylab as plt

def otsu(im):
    histogram,bins = np.histogram(im.flatten(),256,[0,256],density=True)
    P1 = np.zeros(256)
    P2 = np.zeros(256)
    m = np.zeros(256) #
    for i,p_i in enumerate(histogram):
        P_k = np.sum(histogram[0:i+1])
        P1[i] = P_k
        P2[i] = 1-P_k
        i_list = np.arange(i+1)
        m[i] = np.sum(i_list * histogram[:i+1])
    m1 = np.zeros(256)
    m2 = np.zeros(256)
    for i,p_i in enumerate(P1):
        i_list = np.arange(i + 1)
        if p_i != 0:
            sip1 = np.sum(i_list * histogram[0:i+1])
            m1[i] = 1/p_i * sip1
        else:
            m1[i] = np.mean(i_list)
    for i, p_i in enumerate(P2):
        i_list = np.arange(i, 256)
        if p_i != 0:
            sip2 = np.sum(i_list * histogram[i:256])
            m2[i] = (1 / p_i * sip2)
        else:
            m2[i] = np.mean(i_list)
    m_G = m[-1]
    i_list = np.arange(256)
    sigma_G2 =  np.sum((i_list-m_G) ** 2 * histogram)
    sigma_B2_np = np.zeros(256)
    sigma_B2_np = P1 *P2 *((m1-m2) ** 2)
    # print(sigma_B2_np)
    # print(m1,m2)
    # print(P1 + P2)
    T = np.mean(np.argwhere(sigma_B2_np == np.amax(sigma_B2_np)))
    return T

def l_walk(im,half_n= 7):
    w,h = im.shape
    newimg2 = np.zeros((w,h))
    histogram, bins = np.histogram(im.flatten(), 256, [0, 256], density=True)
    i_list = np.arange(256)
    m_G = np.sum(i_list * histogram)
    im_border= cv.copyMakeBorder(im,10,10,10,10,cv.BORDER_CONSTANT,value=int(m_G))
    for x in range(half_n,w + half_n):
        for y in range(half_n,h + half_n):
            T = otsu(im_border[x - half_n:x + half_n,y-half_n:y + half_n ])
            newimg2[x-half_n,y-half_n] = im[x-half_n,y-half_n] > T
    return newimg2

# read picture
img = cv.imread('img1.tif',0)
newimg = np.zeros(img.shape)
print(img.shape, img.dtype, type(img))

# show image
plt.imshow(img,cmap = 'gray')
plt.axis('off')
plt.show()

# global
T = otsu(img)
newimg =  img > T

plt.imshow(newimg,cmap = 'gray')
plt.axis('off')
plt.show()
# local
newimg2 = l_walk(img,half_n=7)
plt.imshow(newimg2,cmap = 'gray')
plt.axis('off')
plt.show()