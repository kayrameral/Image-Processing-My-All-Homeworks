
import numpy as np
import cv2 as cv

img=cv.imread("lena_grayscale_hq.jpg",0)
#a= cv.xfeatures2D.SURF_create() 
sift=cv.SIFT_create()
#surf=cv.xfeatures2d.SURF_create()
orb=cv.ORB_create(nfeatures=1500) 

keyPoints,descriptors=orb.detectAndCompute(img,None)
image=cv.drawKeypoints(img,keyPoints,img)

cv.imshow("sift.jpg",image)
cv.waitKey(0)







