import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image=cv.imread("noisyImage_Gaussian.jpg",0)
image1=cv.imread("noisyImage_Gaussian_01.jpg",0)

def NonLocalMeansFilter(image,h,tWindowSize,sWindowSize):
    H,W=image.shape
    tWindowSizeH=tWindowSize//2
    sWindowSizeH=sWindowSize//2    
    nonLocalMeansFilter=cv.copyMakeBorder(image,tWindowSizeH,tWindowSizeH,tWindowSizeH,tWindowSizeH,cv.BORDER_REPLICATE)    
    for x in range(H):
        for y in range(W):            
            ValuesOfPixels=0
            ValuesOfWeights=0           
            arr1=nonLocalMeansFilter[tWindowSizeH-sWindowSizeH+x:sWindowSizeH+tWindowSizeH+x+1,tWindowSizeH-sWindowSizeH+y :tWindowSizeH+sWindowSizeH+y+1]
            for i in range(x,tWindowSize-sWindowSize+x,1):
                for j in range(y,y+tWindowSize-sWindowSize,1):  
                    arr2 = nonLocalMeansFilter[i:sWindowSize+i,j:sWindowSize+j]  
                    w1=np.exp(-np.sqrt(np.sum(np.square(arr1 - arr2))/(h**2)))
                    ValuesOfWeights+=w1
                    ValuesOfPixels+=w1*nonLocalMeansFilter[sWindowSize+i,sWindowSize+j]
            ValuesOfPixels/=ValuesOfWeights                  
            nonLocalMeansFilter[tWindowSizeH+x,tWindowSizeH+y]=ValuesOfPixels
    return nonLocalMeansFilter[tWindowSizeH:tWindowSizeH+H,tWindowSizeH:tWindowSizeH+W]

Gaussian=cv.GaussianBlur(image,(5,5),0)
openCvNLM = cv.fastNlMeansDenoising(image,0.1,7,5)
myNLM=NonLocalMeansFilter(image,0.1,7,5)
Gaussian1=cv.GaussianBlur(image1,(5,5),0)
openCvNLM1= cv.fastNlMeansDenoising(image1,0.1,7,5)
myNLM1=NonLocalMeansFilter(image1,0.1,7,5)
plt.figure(figsize=(12,8))
plt.subplot(2,4,1)
plt.imshow(Gaussian,cmap='gray')
plt.title("openCVGaussian")
plt.subplot(2,4,2)
plt.imshow(openCvNLM,cmap='gray')
plt.title("openCVNLM")
plt.subplot(2,4,3)
plt.imshow(myNLM,cmap='gray')
plt.title("myNLM")
plt.subplot(2,4,5)
plt.imshow(Gaussian1,cmap='gray')
plt.title("OpenCVGaussian-1")
plt.subplot(2,4,6)
plt.imshow(openCvNLM1,cmap='gray')
plt.title("openCVNLM-1")
plt.subplot(2,4,7)
plt.imshow(myNLM1,cmap='gray')
plt.title("myNLM-1")
plt.show() 