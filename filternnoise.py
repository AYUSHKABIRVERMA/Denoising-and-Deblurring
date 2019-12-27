import cv2 ,copy 
import numpy as np 
import scipy.signal
from scipy import ndimage
from skimage.draw import circle
from scipy import fftpack, signal
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
# from matplotlib import pyplot as plt

def mapfun(img, low, high):
    return (img-low)/(high-low)

def psnr(img1, img2):
    if(img1==img2).all():
        return 100
    mse = np.mean((img1-img2)**2)
    psnr = 20*np.log10(img1.max()/np.sqrt(mse))
    return psnr

def addGaussianNoise(img,mean,var):
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,img.shape).reshape(img.shape)
    noisy_img = np.clip(((img+gauss)*255).astype('uint8'),0,255)
    cv2.imwrite('Gnoise.jpg', noisy_img)
    # cv2.waitKey()
    return noisy_img    


def addGaussianNoise2(img,mean,var):
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,img.shape).reshape(img.shape)
    noisy_img = img+gauss
    return noisy_img, gauss

def addspnoise(img,percent, svsp):
    out_img = np.copy(img)
    
    num_salt = np.ceil(percent * img.size * svsp)
    num_pepper = np.ceil(percent* img.size * (1. - svsp))

    salt_coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in img.shape]
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in img.shape]
    
    out_img[salt_coords] = 1
    out_img[pepper_coords] = 0

    out_img = np.clip(((out_img)*255).astype('uint8'),0,255)    
    cv2.imwrite("S&P.png", out_img)
    # cv2.waitKey()
    return out_img

def meanfilter(img, size):
    filter = (1/size**2) * np.ones((size,size))

    img_fft = scipy.signal.fftconvolve( img, filter, mode='same' )
    img_fft = np.clip(img_fft.astype('uint8'),0,255)
    
    print("Mean: ",size,psnr(img,img_fft))
    cv2.imwrite("meanfilter"+str(size)+".png", img_fft)
    # cv2.waitKey()

def medianfilter(img, size):
    out_img = np.zeros(img.shape)
    temp =[]
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            for k in range(0, size):
                if i+k - (size//2) < 0 or i+k-(size//2)> len(img)-1:
                    for l in range(size):
                        temp.append(0)
                elif j + k - (size//2) < 0 or j + (size//2) > len(img[0]) - 1:
                    temp.append(0)
                else:
                    for l in range(size):
                        temp.append(img[i + k - size//2][j + l - size//2])
            temp.sort()
            out_img[i][j] = temp[len(temp)//2]
            temp =[]
    out_img = np.clip(out_img.astype('uint8'),0,255) 
    
    print("Median: ",size,psnr(img,out_img))
    cv2.imwrite("medianfilter"+str(size)+".png", out_img)
    # cv2.waitKey()
    
def anisotropic(img, iterations):

    lambdaa = 0.14
    kappa = 15
    dd = np.sqrt(2)
    u = copy.deepcopy(img.astype('float64'))

    # 2D finite difference windows
    In = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64)
    Is = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64)
    Ie = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64)
    Iw = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64)
    Ine= np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64)
    Ise= np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64)
    Isw= np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64)
    Inw= np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64)
    windows = [In, In, Ie, Iw, Ine, Ise, Isw, Inw]
    
    for r in range(iterations):
        # approximate gradients
        delt_i = [ ndimage.filters.convolve(u, w) for w in windows ]

        # approximate diffusion function
        # diff = [ 1./(1 + (n/kappa)**2) for n in delt_i]
        diff = [np.exp(- (n/kappa)**2) for n in delt_i]

        # update image
        terms = [diff[i]*delt_i[i] for i in range(4)]
        terms += [(1/(dd**2))*diff[i]*delt_i[i] for i in range(4, 8)]
        u = u + lambdaa*(sum(terms))


    u = np.clip(u.astype('uint8'),0,255)
    cv2.imwrite("anisotropic.png", u)
    # cv2.waitKey()

def NSR(oimg, noise):
    N = np.abs(fftpack.fft2(noise))**2
    S = np.abs(fftpack.fft2(oimg))**2
    NSR = N/S
    # print(NSR)
    return NSR

def getDFB(img, size):
    kernel = np.zeros((size,size), dtype= np.float32)
    circleCenterCoord = size // 2
    circleRadius = circleCenterCoord +1
    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    
    kernel[rr,cc] = circleRadius/ (np.sqrt((circleCenterCoord-rr)**2 + (circleCenterCoord-cc)**2 )+ circleRadius)
    kernel[0,0] =0
    kernel[0,size-1] =0
    kernel[size-1,0] =0
    kernel[size-1, size-1] =0

    kernel = kernel / np.sum(kernel)

    #############################
    sz =(img.shape[0] - kernel.shape[0], img.shape[1] - kernel.shape[1])
    kernel = np.pad(kernel, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)), 'constant')
    kernel = fftpack.ifftshift(kernel)
    return kernel

def addDiskblurandGauss(img,size):

    kernel = np.zeros((size,size), dtype= np.float32)
    circleCenterCoord = size // 2
    circleRadius = circleCenterCoord +1
    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    
    kernel[rr,cc] = 1 #circleRadius/ (np.sqrt((circleCenterCoord-rr)**2 + (circleCenterCoord-cc)**2 )+ circleRadius)
    kernel[0,0] =0
    kernel[0,size-1] =0
    kernel[size-1,0] =0
    kernel[size-1, size-1] =0

    kernel = kernel / np.sum(kernel)
    
    #############################
    sz =(img.shape[0] - kernel.shape[0], img.shape[1] - kernel.shape[1])
    kernel = np.pad(kernel, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)), 'constant')
    # kernel = fftpack.ifftshift(kernel)
    frek = np.array(abs(fftpack.fft2(kernel)))
    frek = fftpack.ifftshift(frek)
    plt.plot(500),plt.imshow(frek, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    # cv2.imshow(frek)
    # cv2.waitKey()
    # #############################
    kernel = fftpack.ifftshift(kernel)
    img_blur = np.real(fftpack.ifft2(fftpack.fft2(img)*fftpack.fft2(kernel)))
    # cv2.imshow("diskbluroriginal.png", img_blur)
    # cv2.waitKey()
    
    img_blur, gauss = addGaussianNoise2(img_blur, 0, 0.000005)
    # cv2.imshow("diskblur.png", img_blur)
    # cv2.waitKey()
    return img_blur, kernel, gauss

    
def wiener(img, h, oimg,noise):
    K = NSR(oimg, noise)
    # K = 0.5 # you need to find this by expectation of noise power spectrum and expectation of image power spectrum
    G = fftpack.fft2(img)
    H = fftpack.fft2(h)
    W = np.conj(H) / (np.multiply(H, np.conj(H))+K)
    S = G*W
    f = np.real(fftpack.ifft2(S))
    freimg = np.array(abs(fftpack.fft2(f)))
        # freimg = fftpack.ifftshift(freimg)
    magnitude_spectrum = 20*np.log(np.abs(freimg))
    magnitude_spectrum = fftpack.ifftshift(magnitude_spectrum)
    plt.plot(500),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude SpectruS'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    cv2.imshow("weiner.png", f)
    cv2.waitKey()

def wiener2(img, h, K):
    # K = 0.5 # you need to find this by expectation of noise power spectrum and expectation of image power spectrum
    G = fftpack.fft2(img)
    H = fftpack.fft2(h)
    W = np.conj(H) / (np.multiply(H, np.conj(H))+K)
    S = G*W
    f = np.real(fftpack.ifft2(S))
    
    cv2.imshow("weinerreal.png", f)
    cv2.waitKey()
    return f
