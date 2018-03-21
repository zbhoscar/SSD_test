#!/usr/bin/env python

import numpy as np  
import cv2
import math  
import caffe
cap=cv2.VideoCapture('/home/zbh/Desktop/Datasets/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')
outfile='/home/zbh/Desktop/test1.avi'
#videoCapture = cv2.VideoCapture(file_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if math.isnan(fps):			# if fps = nan , set it to 25
	fps=25
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'XVID'), fps, size) 
print fps, size
wait = int(1000/fps)
while (True):  
    ret,frame=cap.read()  
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    videoWriter.write(frame)  
    # cv2.imshow("shiyan",frame)
    cv2.waitKey(wait)

    if cv2.waitKey(1)&0xFF==ord('q'):  
        break

# while(cap.isOpened()):  
#     ret, frame = cap.read()  
#     #cv2.imshow('image', frame)  
#     videoWriter.write(frame)
#     k = cv2.waitKey(20)  

#     if (k & 0xff == ord('q')):
#         break  

cap.release()  
videoWriter.release()
cv2.destroyAllWindows()

# import numpy as np  
# import cv2  
# cap=cv2.VideoCapture(0)  

# while (True):  
#     ret,frame=cap.read()  
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
#     cv2.imshow("shiyan",gray)  
#     if cv2.waitKey(1)&0xFF==ord('q'):  
#         break  
# cap.release()  
# cv2.destroyAllWindows()  

import cv2   

l='/home/zbh/Desktop/caffe-ssd/examples/images/fish-bike.jpg'
img = caffe.io.load_image(l)
#img = cv2.imread(l)   
cv2.namedWindow("Image")   

cv2.rectangle(img, (0,0), (100,100),234)   
cv2.imshow("Image", img)
cv2.waitKey (0)  
cv2.destroyAllWindows()  