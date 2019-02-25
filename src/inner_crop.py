#!/usr/bin/env python

import os, argparse, glob
import cv2
import numpy as np

# parse arguments in a beautiful way
# includes automatic help generation
def getArgs():

   # setup parser
   parser = argparse.ArgumentParser(
    description = '''Inner crop algorithm. Removes outer border from.''',
    epilog = '''post bug reports to the github repository''')

   parser.add_argument('-d',
                       '--directory',
                       help = 'location of the data to match',
                       default = '../data/format_1/')

   parser.add_argument('-o',
                       '--output_directory',
                       help = 'location where to store the data',
                       default = '../output/previews/')

   # put arguments in dictionary with
   # keys being the argument names given above
   return parser.parse_args()

def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def innerCrop(image):
    
    # creating copy of original image
    orig = image.copy()
    
    # convert to grayscale (red channel) and blur to smooth
    _,_,gray = cv2.split(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # threshold original image
    ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # find edges
    kernel = np.ones((3,3),np.uint8)
    edged = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, kernel)
    edged = cv2.dilate(edged, kernel, iterations = 3)
    
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    x,y,w,h = cv2.boundingRect(contours[0])
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0)
    
    # get approximate contour
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)
    
        if len(approx) == 4:
            target = approx
            break
    
    # mapping target points original layout
    approx = rectify(target)
    
    # reshuffle coordinates
    x = approx[:,0]
    y = approx[:,1]
    w = int(max(x) - min(x))
    h = int(max(y) - min(y))
    x = int(min(x))
    y = int(min(y))

    # original coordinate references
    pts2 = np.float32([[0,0],
    [image.shape[1],0],
    [image.shape[1],image.shape[0]],
    [0,image.shape[0]]])
    
    # homography calculation
    M = cv2.getPerspectiveTransform(approx, pts2)
    
    # warp perspective
    dst = cv2.warpPerspective(orig, M, None)
    
    return dst

if __name__ == '__main__':
  
  # parse arguments
  args = getArgs()
  
  # list image files to be aligned and subdivided
  files = glob.glob(args.directory + "*.jpg")

  # loop over all files, align and subdivide
  for i, file in enumerate(files):
    
    print("Reading image to align : ", file)
    im = cv2.imread(file)
  
    # extract the prefix of the file under evaluation
    prefix = os.path.basename(file).split('.')[0]
    
    # crop image
    crop = innerCrop(im)
    
    # write to file
    outFilename = "../output/" + prefix + "_crop.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, crop)
