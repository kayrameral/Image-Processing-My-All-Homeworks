import numpy as np
import cv2 as cv

image = cv.imread("lena_grayscale_hq.jpg",0)

def box_filter(img, K_size):
    H, W= img.shape
    # zero padding
    pad = K_size // 2
    box_filter_output = np.zeros((H , W ))
    box_filter_output[ 0:H, 0: W] = img.copy()
    box_filter_output =np.pad(box_filter_output, pad_width=pad)
    tmp = box_filter_output.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            box_filter_output[y, x] = round(np.mean(tmp[y: y + K_size, x: x + K_size]))
    box_filter_output = box_filter_output[0: H, 0: W]
    return box_filter_output

def seperable_filter(img, K_size):
    H, W= img.shape
    # zero padding
    pad = K_size // 2
    seperable_filter_output = np.zeros((H , W ))
    seperable_filter_output[ 0:H, 0: W] = img.copy()
    seperable_filter_output =np.pad(seperable_filter_output, pad_width=pad)
    tmp = seperable_filter_output.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            i=0
            j=0
            y_normalValue=y
            x_normalValue=x
            K_size_array_1=np.zeros((K_size,K_size ))
            K_size_array_2=np.zeros((K_size,K_size ))
            while(i!=K_size):
                K_size_array_1[i,0: K_size]=(((tmp[y,x: x + K_size])))
                K_size_array_1[i,0: K_size] = np.divide(K_size_array_1[i,0: K_size], K_size) 
                i=i+1     
                y=y+1
            y=y_normalValue
            while(j!=K_size):
                K_size_array_2[0:K_size,j]= ((K_size_array_1[0:K_size,j]))
                K_size_array_2[0:K_size,j] = np.divide(K_size_array_2[0:K_size,j], K_size)
                j=j+1     
                x=x+1    
            x=x_normalValue
            seperable_filter_output[y, x] = round(np.sum(K_size_array_2))
    seperable_filter_output = seperable_filter_output[0: H, 0: W]
    return seperable_filter_output


#output_1_x
output_1_1 = box_filter(image, 3)
cv.imwrite("output_1_1.jpg",output_1_1)

output_1_2 = box_filter(image, 11)
cv.imwrite("output_1_2.jpg",output_1_2)

output_1_3 = box_filter(image, 21)
cv.imwrite("output_1_3.jpg",output_1_3)

#output_2_x
output_2_1 = cv.boxFilter(image,0,(3,3),borderType=cv.BORDER_CONSTANT)
cv.imwrite("output_2_1.jpg",output_2_1)
output_2_2 = cv.blur(image,(11,11),borderType=cv.BORDER_CONSTANT)
cv.imwrite("output_2_2.jpg",output_2_2)
output_2_3 = cv.blur(image,(21,21),borderType=cv.BORDER_CONSTANT)
cv.imwrite("output_2_3.jpg",output_2_2)

#output_3_x
output_3_1 = seperable_filter(image, 3)
cv.imwrite("output_3_1.jpg",output_3_1)
output_3_2 = seperable_filter(image, 11)
cv.imwrite("output_3_2.jpg",output_3_2)
output_3_3 = seperable_filter(image, 21)
cv.imwrite("output_3_3.jpg",output_3_3)

#output abs(output_1_x-output_2_x) and abs(output_3_x-output_2_x)
abs_output_1=(abs(output_1_1- output_2_1))
abs_output_2=(abs(output_1_2- output_2_2))
abs_output_3=(abs(output_1_3- output_2_3))
abs_output_4=(abs(output_3_1- output_2_1))
abs_output_5=(abs(output_3_2- output_2_2))
abs_output_6=(abs(output_3_3- output_2_3))

#save abs_output_x
cv.imwrite("abs_output_1.jpg",abs_output_1)
cv.imwrite("abs_output_2.jpg",abs_output_2)
cv.imwrite("abs_output_3.jpg",abs_output_3)
cv.imwrite("abs_output_4.jpg",abs_output_4)
cv.imwrite("abs_output_5.jpg",abs_output_5)
cv.imwrite("abs_output_6.jpg",abs_output_6)

#sum of pixels of absolute difference images
abs_output_1_sum=np.sum(abs_output_1[0:512,0:512])
abs_output_2_sum=np.sum(abs_output_2[0:512,0:512])
abs_output_3_sum=np.sum(abs_output_3[0:512,0:512])
abs_output_4_sum=np.sum(abs_output_4[0:512,0:512])
abs_output_5_sum=np.sum(abs_output_5[0:512,0:512])
abs_output_6_sum=np.sum(abs_output_6[0:512,0:512])

#sum of absolute difference pixels of images
print("abs_output_1 result: " ,abs_output_1_sum)
print("abs_output_2 result: " ,abs_output_2_sum)
print("abs_output_3 result: " ,abs_output_3_sum)
print("abs_output_4 result: " ,abs_output_4_sum)
print("abs_output_5 result: " ,abs_output_5_sum)
print("abs_output_6 result: " ,abs_output_6_sum)

#find max 
if(abs_output_1_sum>abs_output_2_sum and abs_output_1_sum>abs_output_3_sum ):
    print("max abs_output : abs_output_1 " ,abs_output_1_sum)
elif(abs_output_2_sum>abs_output_3_sum and abs_output_2_sum>abs_output_1_sum ):
    print("max abs_output : abs_output_2 " ,abs_output_2_sum)
else :
    print("max abs_output : abs_output_3 " ,abs_output_3_sum)
if(abs_output_4_sum>abs_output_5_sum and abs_output_4_sum>abs_output_6_sum ):
    print("max abs_output : abs_output_4 " ,abs_output_4_sum)
elif(abs_output_5_sum>abs_output_6_sum and abs_output_5_sum>abs_output_4_sum ):
    print("max abs_output : abs_output_5 " ,abs_output_5_sum)
else :
    print("max abs_output : abs_output_6 " ,abs_output_6_sum)

#show the images
output_1_1_show=cv.putText( cv.imread("output_1_1.jpg",0), 'output_1_1', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('output_1_1.jpg',output_1_1_show)
cv.waitKey(0)
output_1_2_show=cv.putText( cv.imread("output_1_2.jpg",0), 'output_1_2', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('output_1_2.jpg',output_1_2_show)
cv.waitKey(0)
output_1_3_show=cv.putText( cv.imread("output_1_3.jpg",0), 'output_1_3', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('output_1_3.jpg',output_1_3_show)
cv.waitKey(0)
output_2_1_show=cv.putText( cv.imread("output_2_1.jpg",0), 'output_2_1', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('output_2_1.jpg',output_2_1_show)
cv.waitKey(0)
output_2_2_show=cv.putText( cv.imread("output_2_2.jpg",0), 'output_2_2', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('output_2_2.jpg',output_2_2_show)
cv.waitKey(0)
output_2_3_show=cv.putText( cv.imread("output_2_3.jpg",0), 'output_2_3', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('output_2_3.jpg',output_2_3_show)
cv.waitKey(0)
output_3_1_show=cv.putText( cv.imread("output_3_1.jpg",0), 'output_3_1', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('output_3_1.jpg',output_3_1_show)
cv.waitKey(0)
output_3_2_show=cv.putText( cv.imread("output_3_2.jpg",0), 'output_3_2', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('output_3_2.jpg',output_3_2_show)
cv.waitKey(0)
output_3_3_show=cv.putText( cv.imread("output_3_3.jpg",0), 'output_3_3', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('output_3_3.jpg',output_3_3_show)
cv.waitKey(0)
abs_output_1_show=cv.putText( cv.imread("abs_output_1.jpg",0), 'abs_output_1', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('abs_output_1.jpg',abs_output_1_show)
cv.waitKey(0)
abs_output_2_show=cv.putText( cv.imread("abs_output_2.jpg",0), 'abs_output_2', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('abs_output_2.jpg',abs_output_2_show)
cv.waitKey(0)
abs_output_3_show=cv.putText( cv.imread("abs_output_3.jpg",0), 'abs_output_3', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('abs_output_3.jpg',abs_output_3_show)
cv.waitKey(0)
abs_output_4_show=cv.putText( cv.imread("abs_output_4.jpg",0), 'abs_output_4', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('abs_output_4.jpg',abs_output_4_show)
cv.waitKey(0)
abs_output_5_show=cv.putText( cv.imread("abs_output_5.jpg",0), 'abs_output_5', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('abs_output_5.jpg',abs_output_5_show)
cv.waitKey(0)
abs_output_6_show=cv.putText( cv.imread("abs_output_6.jpg",0), 'abs_output_6', (50, 50), cv.FONT_HERSHEY_DUPLEX,1.0, (255,0,0), 1,cv.LINE_AA, False)
cv.imshow('abs_output_6.jpg',abs_output_6_show)
cv.waitKey(0)
cv.destroyAllWindows()