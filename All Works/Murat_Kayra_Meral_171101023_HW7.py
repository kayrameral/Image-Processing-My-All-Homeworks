import numpy as np
import cv2 as cv

image1 = cv.imread("noisyImage_Gaussian.jpg",0)
golden = cv.imread("lena_grayscale_hq.jpg",0)
image2 = cv.imread("noisyImage_Gaussian_01.jpg",0)
image1_normalized = cv.normalize(image1, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
image2_normalized = cv.normalize(image2, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

def adaptive_mean_filter(img, K_size):
    pad = K_size // 2
    adaptive_mean_filter_output = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    H, W= adaptive_mean_filter_output.shape
    tmp = adaptive_mean_filter_output.copy()
    for y in range(pad,H):
        for x in range(pad,W):
            average = np.mean(tmp[y-pad:y+pad+1,x-pad:x+pad+1])
            var= np.var(tmp[y-pad:y+pad+1,x-pad:x+pad+1])
            if(var!=0):
                adaptive_mean_filter_output[y,x]=(tmp[y,x]-((0.0042/var)*((tmp[y,x]-average))))
    adaptive_mean_filter_output = adaptive_mean_filter_output[pad: H-pad, pad: W-pad]
    adaptive_mean_filter_output=cv.normalize(adaptive_mean_filter_output, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return adaptive_mean_filter_output.astype(np.uint8)

def adaptive_mean_filter1(img, K_size):
    pad = K_size // 2
    adaptive_mean_filter_output = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    H, W= adaptive_mean_filter_output.shape
    tmp = adaptive_mean_filter_output.copy()
    for y in range(pad,H):
        for x in range(pad,W):
            average = np.mean(tmp[y-pad:y+pad+1,x-pad:x+pad+1])
            var= np.var(tmp[y-pad:y+pad+1,x-pad:x+pad+1])
            if(var!=0):
                adaptive_mean_filter_output[y,x]=(tmp[y,x]-((0.0009/var)*(tmp[y,x]-average)))
    adaptive_mean_filter_output = adaptive_mean_filter_output[pad: H-pad, pad: W-pad]
    adaptive_mean_filter_output=cv.normalize(adaptive_mean_filter_output, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return adaptive_mean_filter_output.astype(np.uint8)





output_1_1 = adaptive_mean_filter(image1_normalized,5)
output_1_2=cv.blur(image1,(3,3),cv.BORDER_REPLICATE)
output_1_3=cv.blur(image1,(5,5),cv.BORDER_REPLICATE)
output_1_4=cv.GaussianBlur(image1,(3,3),0,cv.BORDER_REPLICATE)
output_1_5=cv.GaussianBlur(image1,(5,5),0,cv.BORDER_REPLICATE)
output_1_6=cv.bilateralFilter(image1, 5, 3, 0.9, borderType = cv.BORDER_REPLICATE) 




adaptivemeanfilterPSNR=cv.PSNR(golden,output_1_1)
boxFilterPSNR1=cv.PSNR(golden,output_1_2)
boxFilterPSNR2=cv.PSNR(golden,output_1_3)
gaussianFilterPSNR1=cv.PSNR(golden,output_1_3)
gaussianFilterPSNR2=cv.PSNR(golden,output_1_5)
bilateralFilterPNSR=cv.PSNR(golden,output_1_6)

print('output_1_1_PSNR1: ',adaptivemeanfilterPSNR)
print('output_1_2_PSNR2: ',boxFilterPSNR1)
print('output_1_3_PSNR3: ',boxFilterPSNR2)
print('output_1_4_PSNR4: ',gaussianFilterPSNR1)
print('output_1_5_PSNR5: ',gaussianFilterPSNR2)
print('output_1_6_PSNR6: ',bilateralFilterPNSR)

output_2_1 = adaptive_mean_filter1(image2_normalized,5)
output_2_2=cv.blur(image2,(3,3),cv.BORDER_REPLICATE)
output_2_3=cv.blur(image2,(5,5),cv.BORDER_REPLICATE)
output_2_4=cv.GaussianBlur(image2,(3,3),0,cv.BORDER_REPLICATE)
output_2_5=cv.GaussianBlur(image2,(5,5),0,cv.BORDER_REPLICATE)
output_2_6=cv.bilateralFilter(image2, 3, 0.1, 1, borderType = cv.BORDER_REPLICATE) 

adaptivemeanfilterPSNR_2=cv.PSNR(golden,output_2_1)
boxFilterPSNR1_2=cv.PSNR(golden,output_2_2)
boxFilterPSNR2_2=cv.PSNR(golden,output_2_3)
gaussianFilterPSNR1_2=cv.PSNR(golden,output_2_3)
gaussianFilterPSNR2_2=cv.PSNR(golden,output_2_5)
bilateralFilterPNSR_2=cv.PSNR(golden,output_2_6)



print('output_2_1_PSNR1: ',adaptivemeanfilterPSNR_2)
print('output_2_2_PSNR2: ',boxFilterPSNR1_2)
print('output_2_3_PSNR3: ',boxFilterPSNR2_2)
print('output_2_4_PSNR4: ',gaussianFilterPSNR1_2)
print('output_2_5_PSNR5: ',gaussianFilterPSNR2_2)
print('output_2_6_PSNR6: ',bilateralFilterPNSR_2)