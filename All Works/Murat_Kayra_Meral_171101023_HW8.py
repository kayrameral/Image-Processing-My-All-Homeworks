import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv




def myNLM(image,h,tWindowSize,sWindowSize):
    height,weight=image.shape
    tWindowSizeH=tWindowSize//2
    sWindowSizeH=sWindowSize//2    
    returnValue=cv.copyMakeBorder(image,tWindowSizeH,tWindowSizeH,tWindowSizeH,tWindowSizeH,cv.BORDER_REPLICATE)    
    for x in range(height):
        for y in range(weight):            
            ValuesOfPixels=0
            ValuesOfWeights=0           
            v1=returnValue[x-sWindowSizeH + tWindowSizeH: x+sWindowSizeH + tWindowSizeH +1,y-sWindowSizeH + tWindowSizeH:y + sWindowSizeH + tWindowSizeH+1]
            for i in range(x,x+tWindowSize-sWindowSize,1):
                for j in range(y,y+tWindowSize-sWindowSize,1):  
                    v2 = returnValue[i:i + sWindowSize, j:j + sWindowSize]  
                    w=np.sum(np.square(v1 - v2))                    
                    w/=h**2
                    w=np.exp(-np.sqrt(w))
                    sumOfW+=w
                    pixels+=w*returnValue[i+sWindowSize, j+sWindowSize]
            pixels /=sumOfW                  
            returnValue[x+tWindowSizeH, y+tWindowSizeH]=pixels
    return returnValue[tWindowSizeH:height+tWindowSizeH, tWindowSizeH:weight+tWindowSizeH]

image =cv.imread("noisyImage_Gaussian.jpg",0)
image01 =cv.imread("noisyImage_Gaussian_01.jpg",0)

cvGoussian=cv.GaussianBlur(image,(5,5),0)
cvNLM = cv.fastNlMeansDenoising(image,10,7,5)
myNlm=myNLM(image,10,7,5)
#myNLM ile fastNlMeansDenoising kodlarına aynı girdileri vermek yetiyor
cvGoussian01=cv.GaussianBlur(image01,(5,5),0)
cvNLM01 = cv.fastNlMeansDenoising(image01,10,7,5)
myNlm01=myNLM(image01,10,7,5)
#cok deger verince islem cok uzun suruyor
plt.figure(figsize=(20,12))

plt.subplot(2,4,1)
plt.imshow(cvGoussian,cmap='gray')
plt.title("cv-Goussian")

plt.subplot(2,4,2)
plt.imshow(cvNLM,cmap='gray')
plt.title("cv-NLM")
#+": [PSNR {0:.3f} fark".format(cv.PSNR(myNlm, cvNLM))
plt.subplot(2,4,3)
plt.imshow(myNlm,cmap='gray')
plt.title("myNlm Filter")

plt.subplot(2,4,5)
plt.imshow(cvGoussian01,cmap='gray')
plt.title("cv-Goussian,01" )

plt.subplot(2,4,6)
plt.imshow(cvNLM01,cmap='gray')
plt.title("cv-NLM,01" )

plt.subplot(2,4,7)
plt.imshow(myNlm01,cmap='gray')
plt.title("myNlm,01" )

plt.show() 