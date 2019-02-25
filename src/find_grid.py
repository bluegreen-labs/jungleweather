#!/usr/bin/env python

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from statistics import median

# read image
img = cv2.imread('format_1_6118_004_1_4.jpg',0)

# threshold original image
ret, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# apply directional x/y-axis sobel filter to detect edges
sobelx = cv2.Sobel(bw, cv2.CV_8UC1,1,0,ksize = 3)
sobely = cv2.Sobel(bw, cv2.CV_8UC1,0,1,ksize = 3)

# dilate the edges for a stronger linear signal
kernel = np.ones((3,3),np.uint8)
sobelx = cv2.dilate(sobelx, kernel, iterations = 1)
sobely = cv2.dilate(sobely, kernel, iterations = 1)

# detect lines using a hough transform
minLineLength = 50
maxLineGap = 10
lines_x = cv2.HoughLinesP(sobelx,1,np.pi/180,100,minLineLength,maxLineGap)
lines_y = cv2.HoughLinesP(sobely,1,np.pi/180,100,minLineLength,maxLineGap)

# grab the ones which have little horizontal or vertical
# tolerance in the difference between coordinate pairs
xc = []
yc = []

for line in lines_x:
    for x1,y1,x2,y2 in line:
        if abs(x1 - x2) < 10:
         xc.append(np.mean([x1,x2]))

for line in lines_y:
    for x1,y1,x2,y2 in line:
        if abs(y1 - y2) < 3:
         yc.append(np.mean([y1,y2]))

# sort these coordinates
xc = sorted(xc)
yc = sorted(yc)

# find the maximum value of the sorted data
# to detect the split between coordinates
# !!! this assumes that there are pairs
xc_loc = np.diff(xc)
yc_loc = np.diff(yc)

x_center = img.shape[1]/2
y_center = img.shape[0]/2

x_threshold = 10
y_threshold = 10

if max(xc_loc) < x_threshold:
 x = int(median(xc))
 if x > x_center:
  x1 = 0
  x2 = x
 else:
  x1 = x
  x2 = img.shape[1]-1
else:
 xc_loc = xc_loc.argmax()
 x1 = int(median(xc[0:xc_loc])) + 3 # offset sobel filter
 x2 = int(median(xc[xc_loc+1:])) - 6

if max(yc_loc) < y_threshold:
 y = int(median(yc))
 if y > y_center:
  y1 = y
  y2 = 0
 else:
  y1 = img.shape[0]-1
  y2 = y
else:
 yc_loc = yc_loc.argmax()
 y1 = int(median(yc[0:xc_loc])) + 3 
 y2 = int(median(yc[xc_loc+1:])) - 3

#  print output
print(str(x1) + " " + str(x2) + " " + str(y1) + " " + str(y2))

# create overlay on original image
cv2.line(img,(x1,y1),(x2,y1),(255,255,255),1)
cv2.line(img,(x1,y2),(x2,y2),(255,255,255),1)
cv2.line(img,(x1,y1),(x1,y2),(255,255,255),1)
cv2.line(img,(x2,y1),(x2,y2),(255,255,255),1)

# write to file
cv2.imwrite('Written_Back_Results.jpg',img)

img = img[y1:y2,x1:x2]

# blurring for OTSU thresholding
img = cv2.GaussianBlur(img,(7, 7),0)
    
# threshold original image
ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)



