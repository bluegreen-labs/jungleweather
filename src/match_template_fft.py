#!/usr/bin/env python

# Import necessary libraries.
import os, argparse, glob, tempfile, shutil, warnings
import cv2
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import imreg_dft as ird

# local functions
from inner_crop import *
from label_table_cells import *
from match_preview import *

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
                       
   parser.add_argument('-f',
                       '--file',
                       help = 'file with location of the data to match',
                       default = '../data-raw/format_1/format_1_6118.csv')

   parser.add_argument('-o',
                       '--output_directory',
                       help = 'location where to store the data',
                       default = '/backup/cobecore/zooniverse/format_1_batch_1/')

   parser.add_argument('-s',
                       '--subsets',
                       help = 'create subsets according to a coordinates file')

   parser.add_argument('-sr',
                       '--scale_ratio',
                       help = 'shrink data by factor x, for faster processing',
                       default = 0.2)
                       
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
                       
   parser.add_argument('-c',
                       '--crop',
                       help='crop before binarization',
                       default = False)                    
                       
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

def alignImages(im, template, size):

  im0_orig = cv2.imread(template)
  height, width = im0_orig.shape[:2]
  im1_orig = cv2.imread(im)
  
  size_factor = max(im1_orig.shape)/800
  im0,left,right = resize(im0_orig, size = size)

  # the image to be transformed
  im1,_,_ = resize(im1_orig, size = size)
  im1_orig,_,_ = resize(im1_orig, size = int(max(im1_orig.shape)), flat = False)
  im0_orig,_,_ = resize(im0_orig, size = int(max(im0_orig.shape)))

  # find template match
  result = ird.similarity(im0, im1, numiter=3)

  # transform the image, scaled to the original
  timg = result["timg"]
  timg_scaled = ird.transform_img(im1_orig,
    tvec = size_factor * result["tvec"],
    scale = result["scale"],
    angle = result["angle"])
   
  left = int(left * (timg_scaled.shape[0] / size))
  right = int(right * (timg_scaled.shape[0] / size))
   
  timg_scaled = timg_scaled[:, left:(timg_scaled.shape[1] - right)]

  # resize to fit
  resized = cv2.resize(timg_scaled, (width, height), interpolation = cv2.INTER_AREA)
  return resized

def flatten(im):
    
    # histogram stretching + OTSU thresholding + resizing
    # (on the red channel)
    clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize = (15,15))
    im = clahe.apply(im)
    im = cv2.GaussianBlur(im,(3, 3),0)
    ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return im

def resize(im, size, flat = True):
 
 if flat: 
   # split out red channel
   # shows more even ligthing conditions
   _,_,im = cv2.split(im)
   im = flatten(im)
 
 old_size = im.shape[:2] 
 ratio = float(size)/max(old_size)
 new_size = tuple([int(x*ratio) for x in old_size])

 im = cv2.resize(im, (new_size[1], new_size[0]))

 delta_w = size - new_size[1]
 delta_h = size - new_size[0]
 top, bottom = delta_h//2, delta_h-(delta_h//2)
 left, right = delta_w//2, delta_w-(delta_w//2)

 color = [0, 0, 0]
 new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
     value=color)
     
 return(new_im, int(left), int(right))

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
  header = im[0:y[0],:]
  header = cv2.resize(header, (0,0), fx = 0.5, fy = 0.5)
  
  # split prefix
  prefix_values = prefix.split("_")
   
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
  cnn_value = []
  cnn_label = []
  file_name = []
  row = []
  col = []
  index = []
  img_format = []
  img_nr = []

  # return output
  with tf.Session(graph = graph) as sess:
  
    # loop over all x values
    for i, x_value in enumerate(x[0:(len(x)-1)]):
     for j, y_value in enumerate(y[0:(len(y)-1)]):
                 
      # generates cropped sections based upon
      # row and column locations
      try:
       # provide padding
       col_width = int(round((x[i+1] - x[i])/3))
       row_width = int(round((y[j+1] - y[j])/2))
       
       x_min = int(x[i] - col_width)
       x_max = int(x[i+1] + col_width)
       
       y_min = int(y[j] - row_width)
       y_max = int(y[j+1] + row_width)
  
       # trap end of table issues 
       # (when running out of space)
       if x_max > im.shape[1]:
         x_max = int(im.shape[1])
        
       if y_max > im.shape[0]:
         y_max = int(im.shape[0])
    
       # crop images using preset coordinates both
       # with and without a rectangle
       crop_im = im[y_min:y_max, x_min:x_max]
        
       # populate grayscale image
       tf_im = np.full((crop_im.shape[0],
         crop_im.shape[1],3),255,dtype=np.uint8)
       
       # split out red channel
       _,_,red = cv2.split(crop_im)
         
       for l in range(2):
        tf_im[:,:,l] = red
       
       # TF pre-processing
       tf_im = cv2.resize(tf_im, dsize = (128, 128),
         interpolation = cv2.INTER_CUBIC)
       tf_im = cv2.normalize(tf_im.astype('float'),
          None, -0.5, .5, cv2.NORM_MINMAX)
       tf_im = np.asarray(tf_im)
       tf_im = np.expand_dims(tf_im,axis=0)

       # TF classifier
       results = sess.run(output_operation.outputs[0], {
             input_operation.outputs[0]: tf_im })
       results = np.squeeze(results)
       top = results.argsort()[-5:][::-1]
       
       image_name = prefix + "_" + str(i+1) + "_" + str(j+1) + ".png"

       # if the crop routine didn't fail write to disk
       cv2.imwrite(path + "/cells/" + image_name, crop_im, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            
       # write label output data to vectors
       cnn_value.append(results[top[0]])
       cnn_label.append(labels[top[0]])
       
       # add filename and row / col numbers
       file_name.append(image_name)
       col.append(i + 1)
       row.append(j + 1)
      
       # add nilco index, format and image number
       index.append(prefix_values[1])
       img_format.append(prefix_values[2])
       img_nr.append(prefix_values[3])
      
      except:
       # Continue to next iteration on fail
       # happens when index runs out
       continue
      
  # concat data into pandas data frame
  df = pd.DataFrame({'cnn_label':cnn_label,
                   'cnn_value':cnn_value,
                   'col':col,
                   'row':row,
                   'index':index,
                   'format':img_format,
                   'img_nr':img_nr,
                   'file':file_name})

  # construct path
  out_file = os.path.join(path + "/labels/", prefix + "_labels.csv")
  
  # write data to disk
  df.to_csv(out_file, sep=',', index = False)
  
  return df

def setup_outdir(output_directory):
  
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
  if not os.path.exists(output_directory + "/headers/"):
    os.makedirs(output_directory + "/headers/")
    
  if not os.path.exists(output_directory + "/cells/"):
    os.makedirs(output_directory + "/cells/")
    os.makedirs(output_directory + "/cells/dl/")
   
  if not os.path.exists(output_directory + "/previews/"):
    os.makedirs(output_directory + "/previews/")
  
  if not os.path.exists(output_directory + "/labels/"):
    os.makedirs(output_directory + "/labels/")

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
    
  # list image files to be aligned and subdivided
  files = pd.read_csv(args.file, header=None)
  files = pd.DataFrame(files) 

  # Some verbose feedback before kicking off the processing
  print("\n")
  print("Reading reference image : " + str(args.template))
  print("Sourcing from : " + str(args.file))
  print("Saving to : " + str(args.output_directory))
  print("\n")
  
  # loop over all files, align and subdivide and report progress
  with tqdm(total = files.shape[0], dynamic_ncols=True) as pbar:
    for index in files.itertuples():
    
      i = index.Index
      file = index[1]
    
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
  
      # align images
      try:
        
        im_aligned = alignImages(im = file,
         template = args.template , size = 800)
                    
        # create an alignment preview
        sz = im_aligned.shape
        im_preview = np.full((sz[0],sz[1],3),255, dtype=np.uint8)
        im_preview[:,:,1] = im_aligned[:,:,1]
        im_preview[:,:,2] = template_original
          
      except:
        error_log(args.output_directory, "alignment", file)
        continue      
            
      # setup output directories if required
      setup_outdir(output_directory)
        
      # cutting things up into cookies    
      try:
      
        labels = cookieCutter(
          guides,
          im_aligned,
          output_directory,
          prefix,
          args.graph,
          args.labels)
                      
        # Write aligned image to disk, including markings of
        # which cells were ok or not (resize, compress to reduce size)
        im_preview = print_labels(im_preview, guides, labels)    
        im_preview = cv2.resize(im_preview, (0,0), fx = 0.25, fy = 0.25)
        filename = os.path.join(output_directory + "/previews",
                                prefix + "_preview.jpg")
        cv2.imwrite(filename, im_preview, [cv2.IMWRITE_JPEG_QUALITY, 50])
       
      except:
        error_log(args.output_directory, "label", file)
        continue
