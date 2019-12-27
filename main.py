import cv2 
import numpy as np 
import scipy
from scipy import fftpack
from skimage.draw import circle
from matplotlib import pyplot as plt
from skimage.draw import circle
# from pyblur import *
from filternnoise import addGaussianNoise, addspnoise, meanfilter, medianfilter, anisotropic, addDiskblurandGauss, mapfun, wiener,wiener2, getDFB
    
def Gaussian(img):
    gimg = addGaussianNoise(mapfun(img,img.min(),img.max()),0,0.001) 
    
#     sizes =[3,5,7,9]
#     for size in sizes:
#         meanfilter(gimg, size)
#         medianfilter(gimg, size)
    anisotropic(gimg, 20)


def SaltandPepper(img):
    spimg = addspnoise(mapfun(img,img.min(),img.max()), 0.04, 0.5)
    
#     sizes =[3,5,7,9]
#     for size in sizes:
#         meanfilter(spimg, size)
#         medianfilter(spimg, size)
    anisotropic(spimg, 15)


def DiskBlur(img):
        freimg = np.array(abs(fftpack.fft2(img)))
        # freimg = fftpack.ifftshift(freimg)
        magnitude_spectrum = 20*np.log(np.abs(freimg))
        magnitude_spectrum = fftpack.ifftshift(magnitude_spectrum)
        plt.plot(500),plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()
    
        dimg, h,gauss= addDiskblurandGauss(img,15) 
        out_img = wiener(dimg, h, img, gauss)


if __name__ == "__main__":
    img = cv2.imread('barbara.png',0)

#     cv2.imshow('normal.png', img)
#     cv2.waitKey()
    
#     Gaussian(img)
#     SaltandPepper(img)
    DiskBlur(mapfun(img,img.min(),img.max()))
