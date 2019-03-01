#!/usr/bin/env python

# Import necessary libraries.
import os, argparse, glob, tempfile, shutil, warnings
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# local functions
from inner_crop import *
from label_table_cells import *
from match_preview import *
from flatten_image import *

# set TF log level (suppress verbose output)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# ignore python warnings
warnings.simplefilter("ignore")

# argument parser
def getArgs():

   parser = argparse.ArgumentParser(
    description = '''Table alignment and subsetting script: 
                    Allows for the alignment of scanned tables relative to a
                    provided template. Subsequent cutouts can be made when 
                    providing a csv file with row and columns coordinates.''',
    epilog = '''post bug reports to the github repository''')
    
   parser.add_argument('-t',
                       '--template',
                       help = 'template to match to data',
                       default = '../data-raw/templates/format_1.jpg')
                       
   parser.add_argument('-d',
                       '--directory',
                       help = 'location of the data to match',
                       default = '/scratch/cobecore/tmp/format_1/')

   parser.add_argument('-o',
                       '--output_directory',
                       help = 'location where to store the data',
                       default = '/scratch/cobecore/formatted_scans/format_1/')

   parser.add_argument('-s',
                       '--subsets',
                       help = 'create subsets according to a coordinates file')

   parser.add_argument('-sr',
                       '--scale_ratio',
                       help = 'shrink data by factor x, for faster processing',
                       default = 0.25)
                       
   parser.add_argument('-g',
                       '--graph',
                       help='graph/model to be executed',
                       default = './cnn_model/cnn_graph.pb')
                       
   parser.add_argument('-l',
                       '--labels',
                       help='name of file containing labels',
                       default = './cnn_model/cnn_labels.txt')

   parser.add_argument('-gi',
                       '--guides',
                       help='name of file containing cell guides',
                       default = '../data-raw/templates/guides.txt')

   parser.add_argument('-gm',
                       '--good_match',
                       help='good match percentage for ORB features',
                       default = 0.1)
                       
   parser.add_argument('-mf',
                       '--max_features',
                       help='max number of ORB features to use',
                       default = 15000)
                       
   return parser.parse_args()

def error_log(path, prefix, content):
    filename = os.path.join(path, prefix + "_error_log.txt")
    with open(filename, "a") as text_file:
       text_file.write(content + "\n")

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

def alignImages(im, template, im_original, max_features, good_match):
  
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(max_features)
  keypoints1, descriptors1 = orb.detectAndCompute(im, None)
  keypoints2, descriptors2 = orb.detectAndCompute(template, None)
   
  # Match features
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse = False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * good_match)
  matches = matches[:numGoodMatches]

  # Draw top matches
  #im_matches = cv2.drawMatches(im, keypoints1,
  # template, keypoints2, matches, None)
  
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
  
  # allow reference points only to be 10% off in any direction
  # TODO: create dynamic tolerance parameter - expand to y-values as well!!!
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

def flatten(im):
    
    # histogram stretching + OTSU thresholding + resizing
    # (on the red channel)
    clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize = (15,15))
    im = clahe.apply(im)
    im = cv2.GaussianBlur(im,(7, 7),0)
    ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return im

def cookieCutter(locations,
                 im,
                 path,
                 prefix,
                 model_file,
                 label_file):
  
  # split out the locations
  # convert to integer
  x = np.asarray(locations[0][3], dtype=float)
  x = x.astype(int)
  x = np.sort(x)
  y = np.asarray(locations[0][2], dtype=float)
  y = y.astype(int)
  y = np.sort(y)
  
  # extract header and resize
  header = im[0:(y[0] - 400),:]
  header = cv2.resize(header, (0,0), fx = 0.5, fy = 0.5)
  
  # annotate header
  cv2.line(header, (365, 0), (365, header.shape[1]), 255, 2)
  cv2.putText(header, "A", (375, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
  cv2.line(header, (1040, 0), (1040, header.shape[1]), 255, 2)
  cv2.putText(header, "B", (1050, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
  
  # write header to file
  cv2.imwrite(path + "/headers/" + prefix + "_header.png",
    header,
    [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

  # setup CNN
  # load default settings
  input_height = 128 #224 #299
  input_width = 128 #224 #299
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"

  # these things are static
  graph = load_graph(model_file)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  # load labels
  labels = load_labels(label_file)

  # initiate empty vectores
  cnn_values = []
  cnn_labels = []
  file_names = []

  # return output
  with tf.Session(graph=graph) as sess:
    
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
       #im_rect = im.copy()
       
       # draw rectangle
       #cv2.line(im_rect, (x_min+col_width-15, y_max-row_width+10),
       #(x_min+col_width-5, y_max-row_width+10), 255, 3)
       #cv2.line(im_rect, (x_min+col_width-15, y_max-row_width+10),
       #(x_min+col_width-15, y_max-row_width), 255, 3)
  
       # shaded section
       #im_rect[y_max-row_width+10:,:] = im_rect[y_max-row_width+10:,:] * 0.7
  
       # crop images using preset coordinates both
       # with and without a rectangle
       #crop_im_rect = im_rect[y_min:y_max,x_min:x_max]
       crop_im = im[y_min:y_max,x_min:x_max]
       
      except:
       # Continue to next iteration on fail
       # happens when index runs out
       continue
       
      tf_im = np.full((crop_im.shape[0], 
        crop_im.shape[1],3),255,dtype=np.uint8)
        
      for l in range(2):
       tf_im[:,:,l] = crop_im
        
      tf_im = cv2.resize(tf_im, dsize=(128, 128),
        interpolation = cv2.INTER_CUBIC)
      tf_im = cv2.normalize(tf_im.astype('float'),
         None, -0.5, .5, cv2.NORM_MINMAX)
      tf_im = np.asarray(tf_im)
      tf_im = np.expand_dims(tf_im,axis=0)
       
      results = sess.run(output_operation.outputs[0], {
             input_operation.outputs[0]: tf_im
         })
      results = np.squeeze(results)
      top = results.argsort()[-5:][::-1]
      cnn_values.append(results[top[0]])
      cnn_labels.append(labels[top[0]])
  
      # if the crop routine didn't fail write to disk
      #filename = path + "/" + prefix + "_" + str(i+1) + "_" + str(j+1) + ".jpg"
      #cv2.imwrite(filename, crop_im_rect, [cv2.IMWRITE_JPEG_QUALITY, 50])
      
      # write the clean images to a temporary directory for
      # screening using a DL routine
      image_name = prefix + "_" + str(i+1) + "_" + str(j+1) + ".jpg"
      file_names.append(image_name)
      #cv2.imwrite(path + "/" + image_name, crop_im)

  # concat data into pandas data frame
  df = pd.DataFrame({'cnn_labels':cnn_labels,
                   'cnn_values':cnn_values,
                   'files':file_names})

  # construct path
  out_file = os.path.join(path + "/labels/", prefix + "_labels.csv")
  
  # write data to disk
  df.to_csv(out_file, sep=',', index = False)
  
  return df

if __name__ == '__main__':
  
  # parse arguments
  args = getArgs()
  
  # get scale ratio
  scale_ratio = float(args.scale_ratio)

  # extract filename and extension of the mask
  mask_name, file_extension = os.path.splitext(args.template)
  mask_name = os.path.basename(mask_name)
  
  # load mask file
  try:
   template = cv2.imread(args.template)
  except:
   print("No valid mask image found at:")
   print(args.template)
   exit()

  # load guides
  try:
   guides = load_guides(args.guides, mask_name)
  except:
    print("No valid guides file found at:")
    print(args.guides)
    exit()

  # split out red channel
  # shows more even ligthing conditions
  _,_,template = cv2.split(template)
  
  # create a copy of the original
  template_original = template
  
  # OTSU thresholding + resizing
  template = cv2.GaussianBlur(template,(7, 7),0)
  ret, template = cv2.threshold(template, 0, 255,
    cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  template = cv2.resize(template, (0,0),
   fx = scale_ratio,
   fy = scale_ratio)
  
  # list image files to be aligned and subdivided
  files = sorted(glob.glob(args.directory + "*.jpg"))

  # Some verbose feedback before kicking off the processing
  print("\n")
  print("Reading reference image : " + str(args.template))
  print("Sourcing from : " + str(args.directory))
  print("Saving to : " + str(args.output_directory))
  print("\n")
  
  # loop over all files, align and subdivide and report progress
  with tqdm(total = len(files), dynamic_ncols=True) as pbar:
    for i, file in enumerate(files):
    
      # update progress
      pbar.update()
      
      # compile final output directory name
      archive_id = os.path.basename(file).split('.')[0].split('_')[0]
      output_directory = args.output_directory + "/" + archive_id + "/"
      prefix = mask_name + "_" + os.path.basename(file).split('.')[0]
      
      # read input data
      try:
        im = cv2.imread(file)
      except:
        error_log(args.output_directory, "read", file)
        continue
  
      # crop red channel, reproject original
      # using the same parameters (crop)
      # im = innerCrop(im)
      
      # create a grayscale copy
      # split out red channel for further processing
      im_tmp = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      _,_,im = cv2.split(im)

      # flatten image (binarization)
      im = flatten(im)

      # resize the image to the size of the template
      im = cv2.resize(im, (template.shape[1], template.shape[0]))
 
      # resize the original data to the size of
      # the full original template (required)
      # to be able to rescale to full resolution
      # with simple homography conversion factors
      im_tmp = cv2.resize(im_tmp,
       (template_original.shape[1],
        template_original.shape[0]))
  
      try:
        im_aligned, h = alignImages(im,
                                    template,
                                    im_tmp,
                                    args.max_features,
                                    args.good_match)
      
        # create an alignment preview
        sz = im_aligned.shape
        im_preview = np.full((sz[0],sz[1],3),255, dtype=np.uint8)
        im_preview[:,:,1] = im_aligned
        im_preview[:,:,2] = template_original
    
        # create output directory if required
        if not os.path.exists(output_directory):
          os.makedirs(output_directory)
    
        if not os.path.exists(output_directory + "/headers/"):
          os.makedirs(output_directory + "/headers/")
    
        if not os.path.exists(output_directory + "/cells/"):
          os.makedirs(output_directory + "/cells/")
   
        if not os.path.exists(output_directory + "/previews/"):
          os.makedirs(output_directory + "/previews/")
    
        if not os.path.exists(output_directory + "/labels/"):
          os.makedirs(output_directory + "/labels/")
    
        # cutting things up into cookies
        labels = cookieCutter(guides,
                     im_aligned,
                     output_directory,
                     prefix,
                     args.graph,
                     args.labels)
        
        # Write aligned image to disk, including markings of
        # which cells were ok or not 
        im_preview = print_labels(im_preview, guides, labels)    
        im_preview = cv2.resize(im_preview, (0,0), fx = 0.25, fy = 0.25)
        filename = os.path.join(output_directory + "/previews",
                                prefix + "_preview.jpg")
        cv2.imwrite(filename, im_preview, [cv2.IMWRITE_JPEG_QUALITY, 50])
       
      except:
        error_log(args.output_directory, "alignment", file)
        continue
    
