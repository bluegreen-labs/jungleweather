#!/usr/bin/env python

# Import necessary libraries.
import os, argparse, glob, tempfile, shutil
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from inner_crop import *
from label_image_list import *
from cnn_check import *

# for ORB feature extraction, will move
# to SIFT in later release (more robust)
MAX_FEATURES = 15000
GOOD_MATCH_PERCENT = 0.1

# parse arguments in a beautiful way
# includes automatic help generation
def getArgs():

   # setup parser
   parser = argparse.ArgumentParser(
    description = '''Table alignment and subsetting script: 
                    Allows for the alignment of scanned tables relative to a
                    provided template. Subsequent cutouts can be made when 
                    providing a csv file with row and columns coordinates.''',
    epilog = '''post bug reports to the github repository''')
    
   parser.add_argument('-t',
                       '--template',
                       help = 'template to match to data',
                       default = '../data/templates/format_1.jpg')
                       
   parser.add_argument('-d',
                       '--directory',
                       help = 'location of the data to match',
                       default = '../data/format_1/')

   parser.add_argument('-o',
                       '--output_directory',
                       help = 'location where to store the data',
                       default = '/home/khufkens/Desktop/output/')

   parser.add_argument('-s',
                       '--subsets',
                       help = 'create subsets according to a coordinates file')

   parser.add_argument('-sr',
                       '--scale_ratio',
                       help = 'shrink data by factor x, for faster processing',
                       default = 0.5)

   parser.add_argument('--input_dir', 
                       help='directory with images to process',
                       default = '../data/cnn_data/out_of_sample/')
                       
   parser.add_argument('--graph',
                       help='graph/model to be executed',
                       default = './cnn_model/cnn_graph.pb')
                       
   parser.add_argument('--labels',
                       help='name of file containing labels',
                       default = './cnn_model/cnn_labels.txt')

   # put arguments in dictionary with
   # keys being the argument names given above
   return parser.parse_args()

def load_guides(filename, mask_name):
   # check if the guides file can be read
   # if not return error
   try:
    guides = []
    file = open(u''+filename,'r')
    lines = file.readlines()
    for line in lines:
     if line.find("Guide:" + mask_name) > -1:
       data = line.split('|')
       data[0] = data[0].split(":")
       data[1] = data[1].split(",")
       data[2] = data[2].split(",")
       data[3] = data[3].split(",")
       guides.append(data)
    file.close()
    return guides
   except:
    print("No subset location file found!")
    print("looking for: " + mask_name + ".csv")
    exit()

def alignImages(im, template, im_original):
  
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im, None)
  keypoints2, descriptors2 = orb.detectAndCompute(template, None)
   
  # Match features
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse = False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  #im_matches = cv2.drawMatches(im, keypoints1, template, keypoints2, matches, None)
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  
  # Prune reference points based upon distance between
  # key points. This assumes a fairly good alignment to start with
  # due to the protocol used (location of the sheets)
  p1 = pd.DataFrame(data=points1)
  p2 = pd.DataFrame(data=points2)
  refdist = abs(p1 - p2)
  
  # allow reference points only to be 20% off in any direction
  refdist = refdist < (im.shape[1] * 0.2)
  refdist = refdist.sum(axis = 1) == 2

  points1 = points1[refdist]
  points2 = points2[refdist]

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  
  # correct for scale factor, only works if both the template
  # and the matching image are of the same size
  h = h * [[1,1,1/scale_ratio],[1,1,1/scale_ratio],[scale_ratio,scale_ratio,1]]

  # Use homography to reshape data
  height, width = im_original.shape
  im_registered = cv2.warpPerspective(im_original, h, (width, height))
  
  # return matched image and ancillary data
  return im_registered, h

def cookieCutter(locations, im, path, tmp_path, prefix):
  
  # cuts cookies out of the aligned data
  # for citsci processing
 
  # split out the locations
  # convert to integer
  x = np.asarray(locations[0][3], dtype=float)
  x = x.astype(int)
  x = np.sort(x)
  y = np.asarray(locations[0][2], dtype=float)
  #y = im.shape[0] - y
  y = y.astype(int)
  y = np.sort(y)
  
  # save the header image
  y_header = int(y.min())
  header = im[0:y[0],:]
  cv2.imwrite(path + "/" + prefix + "_header.jpg", header)

  # loop over all x values
  for i, x_value in enumerate(x):
   for j, y_value in enumerate(y):
    try:
    
     # padding
     col_width = int(round((x[i+1] - x[i])/3))
     row_width = int(round((y[i+1] - y[i])/2))
  
     x_min = int(x[i] - col_width)
     x_max = int(x[i+1] + col_width)
     y_min = int(y[j] - row_width)
     y_max = int(y[j+1] + row_width)

     if x_max > int(im.shape[1]):
      x_max = int(im.shape[1])

     # copy
     im_rect = im.copy()
     
     # draw rectangle
     cv2.line(im_rect, (x_min+col_width-15, y_max-row_width+10),
     (x_min+col_width-5, y_max-row_width+10), 255, 3)
     cv2.line(im_rect, (x_min+col_width-15, y_max-row_width+10),
     (x_min+col_width-15, y_max-row_width), 255, 3)

     # shaded section
     #im_rect[y_max-row_width+10:,:] = im_rect[y_max-row_width+10:,:] * 0.7

     # crop images using preset coordinates both
     # with and without a rectangle
     crop_im_rect = im_rect[y_min:y_max,x_min:x_max]
     crop_im = im[y_min:y_max,x_min:x_max]
     
    except:
     # Continue to next iteration on fail
     # happens when index runs out
     continue

    # if the crop routine didn't fail write to disk
    filename = path + "/" + prefix + "_" + str(i+1) + "_" + str(j+1) + ".jpg"
    cv2.imwrite(filename, crop_im_rect, [cv2.IMWRITE_JPEG_QUALITY, 50])
    
    # write the clean images to a temporary directory for
    # screening using a DL routine
    filename = tmp_path + "/" + prefix + "_" + str(i+1) + "_" + str(j+1) + ".jpg"
    cv2.imwrite(filename, crop_im)

  return

if __name__ == '__main__':
  
  # parse arguments
  args = getArgs()
  
  # get scale ratio
  scale_ratio = float(args.scale_ratio)

  # set default guides filename
  guides_file = "../data/templates/guides.txt"

  # extract filename and extension of the mask
  mask_name, file_extension = os.path.splitext(args.template)
  mask_name = os.path.basename(mask_name)

  # Read reference image and convert to grayscale
  print("Reading reference image : ", args.template)
  try:
   template = cv2.imread(args.template)
  except:
   print("No valid mask image found at:")
   print(args.template)
   exit()

  # split out red channel
  _,_,template = cv2.split(template)
  
  # create an original full sized copy of the template
  # other operations are on scaled versions
  template_original = template
  
  # blurring for OTSU thresholding
  template = cv2.GaussianBlur(template,(7, 7),0)
    
  # threshold original image
  ret, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  # resize
  template = cv2.resize(template, (0,0), fx = scale_ratio, fy = scale_ratio)
  
  # list image files to be aligned and subdivided
  files = glob.glob(args.directory + "*.jpg")

  # loop over all files, align and subdivide
  for i, file in enumerate(files):
  
    # extract the prefix of the file under evaluation
    prefix = mask_name + "_" + os.path.basename(file).split('.')[0]
  
    # compile final output directory name
    archive_id = os.path.basename(file).split('.')[0].split('_')[0]
    output_directory = args.output_directory + "/" + archive_id + "/"
    
    # create output directory if required
    if not os.path.exists(output_directory):
     os.makedirs(output_directory)
  
    print("Reading image to align : " + file)
    im = cv2.imread(file)

    # crop image
    print("-- cropping original data")
    im = innerCrop(im)

    # create a copy (not thresholded) / convert to grayscale
    im_tmp = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # split out red channel
    _,_,im = cv2.split(im)

    # histogram stretching
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
    im = clahe.apply(im)

    # blurring for OTSU thresholding
    im = cv2.GaussianBlur(im,(7, 7),0)
    
    # threshold original image
    ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # resize to the size of the scaled template
    im = cv2.resize(im, (template.shape[1], template.shape[0]))
    
    # resize the original data to the size of
    # the full original template (required)
    # to be able to rescale to full resolution
    # with simple homography tweaks
    im_tmp = cv2.resize(im_tmp,
     (template_original.shape[1],
      template_original.shape[0]))

    # Registered image will be restored in imReg.
    # The estimated homography will be stored in h.
    print("-- matching data to template")
    im_aligned, h = alignImages(im, template, im_tmp)
    
    # create an alignment preview
    # stack the template and the aligned image in an
    # RGB image for quick review
    sz = im_aligned.shape
    im_preview = np.full((sz[0],sz[1],3),255, dtype=np.uint8)
    im_preview[:,:,1] = im_aligned
    im_preview[:,:,2] = template_original
    
    # load guides
    guides = load_guides(guides_file, mask_name)
  
    # cutting things up into cookies
    print("-- extracting header and table cell subsets")
    temp_dir = tempfile.mkdtemp()
    
    cookieCutter(guides,
                 im_aligned,
                 output_directory,
                 temp_dir,
                 prefix)
    
    # screen the cookie cutter data (cells) using the
    # machine learning algorithm, write data to output
    # directory
    # label the data
    print("-- CNN labels")
    labels = label_data(temp_dir,
                        args.graph,
                        args.labels,
                        args.output_directory,
                        prefix)
    
    # remove temporary directory
    shutil.rmtree(temp_dir)
 
    # Write aligned image to disk, including markings of
    # which cells were ok or not 
    im_preview = print_labels(im_preview, guides, labels)    
    im_preview = cv2.resize(im_preview, (0,0), fx = 0.25, fy = 0.25)
    filename = os.path.join(output_directory, prefix + "_preview.jpg")
    print("-- saving preview image")
    cv2.imwrite(filename, im_preview)
    
