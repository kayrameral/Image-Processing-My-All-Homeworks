import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#OUTPUT_1
image_1 =cv.imread("test1.jpg",0) #read the image with cv.imread() method
image_1 = np.array(image_1)
hist,bins=np.histogram(image_1.flatten(),256, [0,256])
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_o = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_o,0).astype('uint8')
output_1 = cdf[image_1]
cv.imwrite("output_1.jpg",output_1) #save the image with cv.imwrite() method
img_show_1 = mpimg.imread("output_1.jpg")                              #show the image 
imgplot = plt.imshow(cv.cvtColor(img_show_1, cv.COLOR_BGR2RGB))        #show the image 
plt.title("OUTPUT_1")                                                  #show the image 
plt.show()                                                             #show the image 

#OUTPUT_2
image_2 = cv.imread("test1.jpg",0) #read the image with cv.imread() method
output_2 = cv.equalizeHist(image_2)
cv.imwrite("output_2.jpg",output_2) #save the image with cv.imwrite() method

img_show_2 = mpimg.imread("output_2.jpg")                              #show the image
imgplot = plt.imshow(cv.cvtColor(img_show_2, cv.COLOR_BGR2RGB))        #show the image
plt.title("OUTPUT_2")                                                  #show the image
plt.show()                                                             #show the image

#OUTPUT_3
def make_histogram(img):
    histogram = np.zeros(256, dtype=int)
    for i in range(img.size):
        histogram[img[i]] += 1
    return histogram

def make_cumsum(histogram):
    cumsum = np.zeros(256, dtype=int)
    cumsum[0] = histogram[0]
    for i in range(1, histogram.size):
        cumsum[i] = cumsum[i-1] + histogram[i]
    return cumsum

def min_value(cumsum):
    hmin=cumsum[0]
    for i in range(1, cumsum.size):
        if(hmin>cumsum[i]):
            hmin=cumsum[i]
    return hmin

def make_mapping(cumsum,hmin):
    mapping = np.zeros(256, dtype=int)
    grey_levels = 255
    for i in range(grey_levels+1):
        mapping[i] =  round(((cumsum[i]-hmin))/(IMG_H*IMG_W-hmin)*grey_levels)
    return mapping

def apply_mapping(img, mapping):
    new_image = np.zeros(img.size, dtype=int)
    for i in range(img.size):
        new_image[i] = mapping[img[i]]
    return new_image

image_3 =cv.imread("test1.jpg",0) #read the image with cv.imread() method
IMG_W, IMG_H = image_3.shape
img = np.array(image_3).flatten()
histogram = make_histogram(img)
cumsum = make_cumsum(histogram)
hmin=min_value(cumsum)
mapping = make_mapping(cumsum,hmin)
new_image = apply_mapping(img, mapping)
output_3 = np.array(np.uint8(new_image.reshape((IMG_H, IMG_W))))
cv.imwrite("output_3.jpg",output_3) #save the image with cv.imwrite() method
img_show_3 = mpimg.imread("output_3.jpg")                             #show the image
imgplot = plt.imshow(cv.cvtColor(img_show_3, cv.COLOR_BGR2RGB))       #show the image
plt.title("OUTPUT_3")                                                 #show the image
plt.show()                                                            #show the image

#ABS_OUTPUT_1 AND ABS_OUTPUT_2
abs_output_1=(abs(output_1- output_2)) #(abs(output_1- output_2))
cv.imwrite("abs_output_1.jpg",abs_output_1)
abs_output_2=(abs(output_2- output_3)) #(abs(output_2- output_3))
cv.imwrite("abs_output_2.jpg",abs_output_2)

#show the abs_output_1 and abs_output_2
imgplot = plt.imshow(cv.cvtColor(abs_output_1, cv.COLOR_BGR2RGB))
plt.title("ABS_OUTPUT_1")
plt.show()
imgplot = plt.imshow(cv.cvtColor(abs_output_2, cv.COLOR_BGR2RGB))
plt.title("ABS_OUTPUT_2")
plt.show()

#sum of pixels of absolute difference images
SumOfAllDifference_1=np.sum(abs_output_1[0:256, 0:256])
SumOfAllDifference_2=np.sum(abs_output_2[0:256, 0:256])
print("abs(output_1-output_2) sum of image pixels result: " , SumOfAllDifference_1)
print("abs(output_2-output_3) sum of image pixels result: " , SumOfAllDifference_2)


