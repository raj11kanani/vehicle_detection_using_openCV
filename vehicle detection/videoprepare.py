# import the necessary packages

import cv2
import imutils
import os
import re
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

col_frames = os.listdir('sample_frame/')
# print(type(col_frames))
# print(col_frames)
# sort file names
col_frames.sort(key=lambda f: int(re.sub('\D', "",f)))


col_images=[]

for i in col_frames:
    # read the frames
    img = cv2.imread('sample_frame/'+i)
    # append the frames to the list
    col_images.append(img)




# kernel for image dilation
kernel = np.ones((4,4),np.uint8)

# font style
font = cv2.FONT_HERSHEY_SIMPLEX

# directory to save the ouput frames
pathIn = "sample_contour_frame/"

for i in range(len(col_images)-1):
    
    # frame differencing
    grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)
    diff_image = cv2.absdiff(grayB, grayA)
    
    # image thresholding
    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
    
    # image dilation
    dilated = cv2.dilate(thresh,kernel,iterations = 1)
    
    # find contours
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    # shortlist contours appearing in the detection zone
    valid_cntrs = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if (x <= 1200) & (y >= 360) & (cv2.contourArea(cntr) >= 25):
            if (y >= 650) & (cv2.contourArea(cntr) < 40):
                break
            valid_cntrs.append(cntr)
            
    # add contours to original frames
    dmy = col_images[i].copy()
    cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)
    
    cv2.putText(dmy, "vehicles detected: " + str(len(valid_cntrs)), (55, 15), font, 1.0, (0, 180, 0), 2)
    cv2.line(dmy, (0, 360),(1280,360),(100, 255, 255))
    cv2.imwrite(pathIn+str(i)+'.png',dmy)  









# specify video name
pathOut = 'sample_vehicle_detection_2.mp4'

# specify frames per second
fps = 29.97
# Next, we will read the final frames in a list:

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]


files.sort(key=lambda f: int(re.sub('\D', '', f)))

for i in range(len(files)):
    filename=pathIn + files[i]
    
    #read frames
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)


out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])

out.release()