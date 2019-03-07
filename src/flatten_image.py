#!/usr/bin/env python

import os, argparse, glob
import cv2
import numpy as np
from inner_crop import *

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

def flatten(im):
    
    # histogram stretching + OTSU thresholding + resizing
    # on the red channel
    clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize = (15,15))
    im = clahe.apply(im)
    im = cv2.GaussianBlur(im,(7, 7),0)
    ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return im

if __name__ == '__main__':
  
  # parse arguments
  args = getArgs()
  
  # list image files to flatten
  files = glob.glob(args.directory + "*.jpg")

  # loop over all files, align and subdivide
  for i, file in enumerate(files):
    
    print("Reading image to align : " + file)
    im = cv2.imread(file)
  
    # extract the prefix of the file under evaluation
    prefix = os.path.basename(file).split('.')[0]
    
    # crop image
    crop = flatten(im)
    
    # write to file
    outFilename = os.path.join(args.output_directory, prefix + "_flatten.jpg")
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, crop)
