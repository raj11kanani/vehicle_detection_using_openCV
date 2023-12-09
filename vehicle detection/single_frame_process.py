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

# empty list to store the frames
col_images=[]

for i in col_frames:
    # read the frames
    img = cv2.imread('sample_frame/'+i)
    # append the frames to the list
    col_images.append(img)

i = 422
print(col_images[0].shape)
print(type(col_images[0].shape))
print(type(col_images[0]))
# for frame in [i, i+1]:
#     plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
#     plt.title("frame: "+str(frame))
#     plt.show()


# convert the frames to grayscale
grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)
# cv2.imshow('grayA',grayA)
# cv2.imshow('grayB',grayB)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plot the image after frame differencing
plt.imshow(cv2.absdiff(grayB, grayA), cmap = 'gray')
plt.show()


diff_image = cv2.absdiff(grayB, grayA)

# perform image thresholding
ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

# plot image after thresholding
plt.imshow(thresh, cmap = 'gray')
plt.show()


# apply image dilation
kernel = np.ones((3,3),np.uint8)
dilated = cv2.dilate(thresh,kernel,iterations = 4)

# plot dilated image
plt.imshow(dilated, cmap = 'gray')
plt.show()

# plot vehicle detection zone
plt.imshow(dilated)
# cv2.line(dilated, (0, 80),(256,80),(100, 0, 0))
cv2.line(dilated, (0, 360),(1280,360),(100, 0, 0))
plt.show()



contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
valid_cntrs = []

for i,cntr in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cntr)
    if (x <= 1200) & (y >= 360) & (cv2.contourArea(cntr) >= 25):
        valid_cntrs.append(cntr)

# count of discovered contours        
len(valid_cntrs)

dmy = col_images[422].copy()

cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)
cv2.line(dmy, (0, 360),(1280,360),(100, 255, 255))
plt.imshow(dmy)
plt.show()
