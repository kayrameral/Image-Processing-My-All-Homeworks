import numpy as np
import cv2 as cv

image1 = cv.imread("noisyImage_Gaussian.jpg",0)
image2 = cv.imread("noisyImage_SaltPepper.jpg",0)
golden = cv.imread("lena_grayscale_hq.jpg",0)
image1_normalized = cv.normalize(image1, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

#====================QUESTION 1======================================
def adaptive_mean_filter(img, K_size):
    pad = K_size // 2
    adaptive_mean_filter_output = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REPLICATE)#BORDER.REFLECT PSNR=26.113, BORDER.CONSTANT PSNR=25, BORDER.REPLICATED PSNR=26.112
    H, W= adaptive_mean_filter_output.shape
    tmp = adaptive_mean_filter_output.copy()
    for y in range(pad,H):
        for x in range(pad,W):
            average = np.mean(tmp[y-pad:y+pad+1,x-pad:x+pad+1])
            var= np.var(tmp[y-pad:y+pad+1,x-pad:x+pad+1])
            adaptive_mean_filter_output[y,x]=(tmp[y,x]-((0.004/var)*(tmp[y,x]-average)))
    adaptive_mean_filter_output = adaptive_mean_filter_output[pad: H-pad, pad: W-pad]
    adaptive_mean_filter_output=cv.normalize(adaptive_mean_filter_output, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return adaptive_mean_filter_output.astype(np.uint8)

#====================QUESTION 2======================================
def LevelALevelBTable(img, y, x, Sxy, Sxymax):
    pad = Sxy // 2
    kernel=img[x-pad:x+pad+1,y-pad:y+pad+1] 
    Zxy=img[x,y]
    Zmed=np.median(kernel)
    Zmax=np.max(kernel)
    Zmin=np.min(kernel)
    if(Zmax>Zmed) and (Zmed>Zmin): 
        if(Zxy>Zmin) and (Zmax>Zxy):
            return Zxy
        else:
            return Zmed
    else:                                
        Sxy=Sxy+2
        if Sxymax>Sxy :
            return LevelALevelBTable(img, y, x, Sxy, Sxymax)
        else:
            return Zmed

def Adaptive_Median_Filter(img, Sxy, Sxymax):
    pad = Sxymax // 2
    img= cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REFLECT)
    H, W= img.shape
    tmp=img.copy()
    for x in range(pad,H):
        for y in range(pad,W):
            tmp[x,y] = LevelALevelBTable(img, y, x, Sxy, Sxymax)
    adaptive_median_filter_output = tmp[pad: H-pad, pad: W-pad]
    return adaptive_median_filter_output.astype(np.uint8) 

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
            new_array=np.append(new_array,[center_value]*(K_size)) #add center value
            new_array.sort() #sort
            center_weighted_median_filter_output[y, x] = (np.median(new_array)) #find median in sorted array
    center_weighted_median_filter_output = center_weighted_median_filter_output[0: H, 0: W] #save
    return center_weighted_median_filter_output.astype(np.uint8)


output_1_1 = adaptive_mean_filter(image1_normalized,5)
output_1_2=cv.blur(image1,(5,5))
output_1_3=cv.GaussianBlur(image1,(5,5),0)
adaptivemeanfilterPSNR=cv.PSNR(golden,output_1_1)
boxFilterPSNR=cv.PSNR(golden,output_1_2)
gaussianFilterPSNR=cv.PSNR(golden,output_1_3)
print('output_1_1_PSNR: ',adaptivemeanfilterPSNR)
print('output_1_2_PSNR: ',boxFilterPSNR)
print('output_1_3_PSNR: ',gaussianFilterPSNR)
output_2_1=Adaptive_Median_Filter(image2,3,7)
output_2_2=cv.medianBlur(image2,3)
output_2_3=cv.medianBlur(image2,5)
output_2_4=cv.medianBlur(image2,7)
output_2_5=center_weighted_median_filter(image2,3)
output_2_6=center_weighted_median_filter(image2,5)
output_2_7=center_weighted_median_filter(image2,7)
output_2_1_PSNR=cv.PSNR(golden,output_2_1)
output_2_2_PSNR=cv.PSNR(golden,output_2_2)
output_2_3_PSNR=cv.PSNR(golden,output_2_3)
output_2_4_PSNR=cv.PSNR(golden,output_2_4)
output_2_5_PSNR=cv.PSNR(golden,output_2_5)
output_2_6_PSNR=cv.PSNR(golden,output_2_6)
output_2_7_PSNR=cv.PSNR(golden,output_2_7)
print('output_2_1_PSNR: ',output_2_1_PSNR)
print('output_2_2_PSNR: ',output_2_2_PSNR)
print('output_2_3_PSNR: ',output_2_3_PSNR)
print('output_2_4_PSNR: ',output_2_4_PSNR)
print('output_2_5_PSNR: ',output_2_5_PSNR)
print('output_2_6_PSNR: ',output_2_6_PSNR)
print('output_2_7_PSNR: ',output_2_7_PSNR)
cv.imwrite('output_1_1.jpg',output_1_1)
cv.imshow('output_1_1.jpg',output_1_1)
cv.imwrite('output_1_2.jpg',output_1_2)
cv.imshow('output_1_2.jpg',output_1_2)
cv.imwrite('output_1_3.jpg',output_1_3)
cv.imshow('output_1_3.jpg',output_1_3)
cv.imwrite('output_2_1.jpg',output_2_1)
cv.imshow('output_2_1.jpg',output_2_1)
cv.imwrite('output_2_2.jpg',output_2_2)
cv.imshow('output_2_2.jpg',output_2_2)
cv.imwrite('output_2_3.jpg',output_2_3)
cv.imshow('output_2_3.jpg',output_2_3)
cv.imwrite('output_2_4.jpg',output_2_4)
cv.imshow('output_2_4.jpg',output_2_4)
cv.imwrite('output_2_5.jpg',output_2_5)
cv.imshow('output_2_5.jpg',output_2_5)
cv.imwrite('output_2_6.jpg',output_2_6)
cv.imshow('output_2_6.jpg',output_2_6)
cv.imwrite('output_2_7.jpg',output_2_7)
cv.imshow('output_2_7.jpg',output_2_7)
cv.waitKey(0)






