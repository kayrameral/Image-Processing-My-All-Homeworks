import numpy as np
import cv2 as cv

image = cv.imread("noisyImage.jpg",0)
golden = cv.imread("lena_grayscale_hq.jpg",0)

#-----------------------------QUESTION 1---------------------------------------------------
def median_filter(img, K_size):
    H, W= img.shape
    pad = K_size // 2
    median_filter_output =cv.copyMakeBorder(img,pad,pad,pad,pad, cv.BORDER_REPLICATE)
    tmp = median_filter_output.copy()
    for y in range(H):
        for x in range(W):
            median_filter_output[y, x] = (np.median(tmp[y: y + K_size, x: x + K_size]))#find median
    median_filter_output = median_filter_output[0: H, 0: W]
    return median_filter_output.astype(np.uint8)

#-----------------------------QUESTION 2---------------------------------------------------
output_box=cv.blur(image, (5,5))#2
output_Gaussian=cv.GaussianBlur(image,(7,7),0)#3
output_2_openCV_medianFilter=cv.medianBlur(image, 5,cv.BORDER_REPLICATE)#4
output_box_filter_PSNR=cv.PSNR(golden,output_box)#2
output_Gaussian_filter_PSNR=cv.PSNR(golden,output_Gaussian)#3
output_median_filter_opencv_PSNR=cv.PSNR(golden,output_2_openCV_medianFilter)#4

#-----------------------------QUESTION 3---------------------------------------------------
def center_weighted_median_filter(img, K_size):
    H, W= img.shape
    pad = K_size // 2
    center_weighted_median_filter_output =cv.copyMakeBorder(img,pad,pad,pad,pad, cv.BORDER_REPLICATE)
    tmp = center_weighted_median_filter_output.copy()
    for y in range(H):
        for x in range(W):
            new_array=tmp[y: y + K_size, x: x + K_size]
            new_array = new_array.flatten() #2d to 1d
            center_value=new_array[len(new_array)//2] #find center pixel
            new_array=np.append(new_array,[center_value,center_value]) #add center value twice
            new_array.sort() #sort
            center_weighted_median_filter_output[y, x] = (np.median(new_array)) #find median in sorted array
    center_weighted_median_filter_output = center_weighted_median_filter_output[0: H, 0: W] #save
    return center_weighted_median_filter_output.astype(np.uint8)
#-------------------------------QUESTION 4--------------------------------------------------

def PSNR(img): #center weighted median filter outputundan düşük bir PSNR değerine sahip olmasına rağmen daha görece daha iyi sonuç veriyor. PSNR değeri iki resim arasındaki her pixelin farklarının karesini alıyor.
    H, W= img.shape #Ondan dolayı uç pixellerden değer seçiyoruz ve tam zıttına bir değer atıyoruz ki farklarının karesi çok çıksın, PSNR düşsün. Bunun amacı az sayıda pixel değişsin ki resim çok bozulmasın.
    tmp = img.copy() #Aynı zamanda da PSNR düşsün.
    for y in range(H):
        for x in range(W):
            if img[y,x]>226: #iki taraftan da uç 29 pixele bakıyoruz ki en az zararla en çok PSNR'ı elde edelim.
                tmp[y,x]=0
            if img[y,x]<29:
                tmp[y,x]=255
    return tmp.astype(np.uint8)
#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
#called method
output_median_filter=median_filter(image, 5)
output_center_weighted_median_filter=center_weighted_median_filter(image, 5)
Low_PSNR=PSNR(golden)

#-------------------------------------------------------------------------------------------------
#sum of absolute difference pixels of images
abs_output_1_sum=np.sum((abs(output_median_filter- output_2_openCV_medianFilter)))
print()
print(" abs_output_1_sum result: " ,abs_output_1_sum)

#-------------------------------------------------------------------------------------------------
#show and write
cv.imwrite('my_output_median_filter.jpg',output_median_filter)
cv.imshow('my_output_median_filter.jpg',output_median_filter)
cv.imwrite('output_box_filter.jpg',output_box)
cv.imshow('output_box_filter.jpg',output_box)
cv.imwrite('output_Gaussian_filter.jpg',output_Gaussian)
cv.imshow('output_Gaussian_filter.jpg',output_Gaussian)
cv.imwrite("output_2_openCV_medianFilter.jpg",output_2_openCV_medianFilter)
cv.imshow('output_2_openCV_medianFilter.jpg',output_2_openCV_medianFilter)
cv.imwrite('output_center_weighted_median_filter.jpg',output_center_weighted_median_filter)
cv.imshow('output_center_weighted_median_filter.jpg',output_center_weighted_median_filter)
cv.imwrite('Low_PSNR_MORE_HIGH_QUALITY.jpg',Low_PSNR)
cv.imshow('Low_PSNR_MORE_HIGH_QUALITY.jpg',Low_PSNR)

#-------------------------------------------------------------------------------------------------
#print PSNR VALUES
print(
    "\n My_median_filter_PSNR: ",cv.PSNR(golden,output_median_filter)
    ,"\n Box_filter_PSNR: ",output_box_filter_PSNR
    ,"\n Gaussian_filter_PSNR: ",output_Gaussian_filter_PSNR
    ,"\n OPENCV_Median_filter_PSNR: ",output_median_filter_opencv_PSNR
    ,"\n Center_Weighted_Median_filter_PSNR: ",cv.PSNR(golden,output_center_weighted_median_filter)
    ,"\n Low_PSNR_MORE_HIGH_QUALITY: ",cv.PSNR(golden,Low_PSNR))

cv.waitKey(0)
