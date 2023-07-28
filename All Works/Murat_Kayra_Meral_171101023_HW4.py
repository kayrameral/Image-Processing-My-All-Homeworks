
import numpy as np
import cv2 as cv

image = cv.imread("lena_grayscale_hq.jpg",0)

def integralImage(img):
    H, W= img.shape
    integral_Image=np.zeros((H+1,W+1),dtype=int)
    tmp=np.zeros((H+1,W+1),dtype=int)
    tmp=img.copy()
    for y in range(H+1):
        for x in range(W+1):
            integral_Image[y, x] =round(np.sum(tmp[0:y,0:x]))
    return integral_Image[1:y+1,1:x+1]

my_integralImage=integralImage(image)
OpenCV_integral=cv.integral(image)
OpenCV_integral=OpenCV_integral[1:,1:]
sum_difference_pixel_1=cv.copyMakeBorder(abs(my_integralImage-OpenCV_integral),0,0,0,0,borderType=cv.BORDER_CONSTANT).astype(np.uint8)
print('sum difference pixels',100*np.sum(abs(OpenCV_integral-my_integralImage)))

def IntegralImage_BoxFilter(img,K_size):
    H, W= img.shape
    pad = K_size // 2
    tmp =cv.copyMakeBorder(img,pad+1,0,pad+1,0,borderType = cv.BORDER_CONSTANT)
    tmp=cv.copyMakeBorder(tmp, 0, pad, 0, pad, borderType = cv.BORDER_REPLICATE)
    IntegralImage_BoxFilterr=np.zeros((H,W))
    for y in range(H):
        for x in range(W):
                sum=(((tmp[y+pad+pad+1,x+pad+pad+1])+(tmp[y,x])-(tmp[y,x+pad+pad+1])-(tmp[y+pad+1+pad,x]))/(K_size*K_size))
                IntegralImage_BoxFilterr[y,x] =sum.round()
    return IntegralImage_BoxFilterr.astype(np.uint8)

My_Integral_Image=IntegralImage_BoxFilter(my_integralImage,3)
OpenCV_box_filter= cv.boxFilter(image,0,(3,3),borderType=cv.BORDER_CONSTANT)
sum_difference_pixel_2=cv.copyMakeBorder(abs(My_Integral_Image-OpenCV_box_filter),0,0,0,0,borderType=cv.BORDER_CONSTANT)
print('sum difference pixels',np.sum(abs(My_Integral_Image-OpenCV_box_filter)))

cv.imwrite('my_integralImage.jpg',my_integralImage)
#cv.imshow('my_integralImage.jpg',my_integralImage)
cv.imwrite('OpenCV_integral.jpg',OpenCV_integral)
#cv.imshow('OpenCV_integral.jpg',OpenCV_integral)
cv.imwrite('sum_difference_pixel_1.jpg',sum_difference_pixel_1)
cv.imshow('sum_difference_pixel_1.jpg',sum_difference_pixel_1)
cv.imwrite('My_Integral_Image.jpg',My_Integral_Image)
cv.imshow('My_Integral_Image.jpg',My_Integral_Image)
cv.imwrite('OpenCV_box_filter_used_integral_image.jpg',OpenCV_box_filter)
cv.imshow('OpenCV_box_filter_used_integral_image.jpg',OpenCV_box_filter)
cv.imwrite('sum_difference_pixel_2.jpg',sum_difference_pixel_2)
cv.imshow('sum_difference_pixel_2.jpg',sum_difference_pixel_2)
cv.waitKey(0)
