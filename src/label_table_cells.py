#!/usr/bin/env python

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Changes made by Koen Hufkens (2018) in order to allow for list of images to be
# used as data source rather than image directories.
# 
# These changes require additional libraries (pandas) to be installed.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, glob, time
import numpy as np
import tensorflow as tf
import pandas as pd
import os.path

def getArgs():
 
  # setup parser
  parser = argparse.ArgumentParser(
    description = '''Read in DL model parameters...''')
  parser.add_argument(
        "--input_dir",
        help="directory with images to process",
        default = "../data/cnn_data/out_of_sample/")
  parser.add_argument(
        "--graph",
        help="graph/model to be executed",
        default = "./cnn_model/cnn_graph.pb")
  parser.add_argument(
        "--labels",
        help="name of file containing labels",
        default = "./cnn_model/cnn_labels.txt")
  parser.add_argument(
        "--output_dir",
        help="name of output dir",
        default = "./")
  return parser.parse_args()

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()
  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph

def read_tensor_from_image_file(file_name,
                                sess,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  #sess = tf.Session()
  result = sess.run(normalized)
  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def label_data(input_dir,
               model_file,
               label_file,
               output_dir,
               prefix):

  # list files
  file_names = glob.glob(input_dir + "/*.jpg")

  # load default settings
  input_height = 299
  input_width = 299
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

  # return output
  with tf.Session(graph=graph) as sess:
   for file_name in file_names:

     # dynamic component
     t = read_tensor_from_image_file(
         file_name,
         sess,
         input_height=input_height,
         input_width=input_width,
         input_mean=input_mean,
         input_std=input_std)

     results = sess.run(output_operation.outputs[0], {
           input_operation.outputs[0]: t
       })
     results = np.squeeze(results)
     top_k = results.argsort()[-5:][::-1]
     cnn_values.append(results[top_k[0]])
     cnn_labels.append(labels[top_k[0]])

  # concat data into pandas data frame
  df = pd.DataFrame({'cnn_labels':cnn_labels,
                   'cnn_values':cnn_values,
                   'files':file_names})

  # construct path
  out_file = os.path.join(output_dir, prefix + "_cnn_labels.csv")

  # write data to disk
  df.to_csv(out_file, sep=',', index = False)
  
  # return dataframe
  return df

if __name__ == "__main__":
  
  # parse arguments
  args = getArgs()

  # set fixed prefix for command line runs
  prefix = "cli_run"

  # label the data
  label_data(args.input_dir,
             args.graph,
             args.labels,
             args.output_dir,
             prefix)

