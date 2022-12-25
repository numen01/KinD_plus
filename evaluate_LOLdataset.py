# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
import argparse

# Define an ArgumentParser object to parse command-line arguments
parser = argparse.ArgumentParser(description='')

# Define command-line arguments for the directories for testing inputs and outputs
parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./LOLdataset/eval15/', help='directory for testing inputs')

# Parse the command-line arguments
args = parser.parse_args()

# Create a TensorFlow session
sess = tf.Session()
# Define a placeholder for a boolean value indicating whether the model is being used for training or evaluation
# The shape of the placeholder is set to an empty tuple to allow for a scalar value
training = tf.placeholder_with_default(False, shape=(), name='training')
# Define placeholders for the model's input data
input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')
input_low_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_r')
input_low_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
input_low_i_ratio = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i_ratio')

# Decompose the input image into reflectance and illumination components
[R_decom, I_decom] = DecomNet(input_decom)
# Store the reflectance and illumination components as the model's output
decom_output_R = R_decom
decom_output_I = I_decom
# Predict the high-resolution reflectance image given the low-resolution reflectance and illumination images
output_r = Restoration_net(input_low_r, input_low_i, training)
# Predict the high-resolution illumination image given the low-resolution illumination image and the low-resolution illumination ratio image
output_i = Illumination_adjust_net(input_low_i, input_low_i_ratio)

# load pretrained model parameters
# Get a list of the trainable variables in the DecomNet model
var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
# Get a list of the trainable variables in the I_enhance_Net model
var_adjust = [var for var in tf.trainable_variables() if 'I_enhance_Net' in var.name]
# Get a list of the trainable variables in the Denoise_Net model
var_restoration = [var for var in tf.trainable_variables() if 'Denoise_Net' in var.name]
# Get a list of all global variables
g_list = tf.global_variables()
# Get a list of the batch normalization moving mean and variance variables
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
# Add the batch normalization variables to the list of variables for the restoration model
var_restoration += bn_moving_vars

# Create a Saver object to restore the variables of the DecomNet model
saver_Decom = tf.train.Saver(var_list = var_Decom)
# Create a Saver object to restore the variables of the I_enhance_Net model
saver_adjust = tf.train.Saver(var_list=var_adjust)
# Create a Saver object to restore the variables of the Denoise_Net model
saver_restoration = tf.train.Saver(var_list=var_restoration)

# Set the directory where the checkpoint file for the DecomNet model is stored
decom_checkpoint_dir ='./checkpoint/decom_model/'
# Try to restore the variables of the DecomNet model from the checkpoint file
ckpt_pre=tf.train.get_checkpoint_state(decom_checkpoint_dir)
if ckpt_pre:
    # If the checkpoint file is found, restore the variables and print a message
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    # If the checkpoint file is not found, print a message
    print('No decomnet pretrained model!')

# Set the directory where the checkpoint file for the I_enhance_Net model is stored
checkpoint_dir_adjust = './checkpoint/illu_model/'
# Try to restore the variables of the I_enhance_Net model from the checkpoint file
ckpt_adjust=tf.train.get_checkpoint_state(checkpoint_dir_adjust)
if ckpt_adjust:
    # If the checkpoint file is found, restore the variables and print a message
    print('loaded '+ckpt_adjust.model_checkpoint_path)
    saver_adjust.restore(sess,ckpt_adjust.model_checkpoint_path)
else:
    # If the checkpoint file is not found, print a message
    print("No adjust net pretrained model!")

# Set the directory where the checkpoint file for the Denoise_Net model is stored
checkpoint_dir_restoration = './checkpoint/restoration_model/'
# Try to restore the variables of the Denoise_Net model from the checkpoint file
ckpt=tf.train.get_checkpoint_state(checkpoint_dir_restoration)
if ckpt:
    # If the checkpoint file is found, restore the variables and print a message
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restoration.restore(sess,ckpt.model_checkpoint_path)
else:
    # If the checkpoint file is not found, print a message
    print("No restoration net pretrained model!")

###load eval data
# Define an empty list to store the low-resolution images
eval_low_data = []
# Define an empty list to store the names of the low-resolution images
eval_img_name =[]
# Get a list of the filenames of the low-resolution images in the 'low' subdirectory of the test directory
eval_low_data_name = glob(args.test_dir + '/low/*.png')
# Sort the list of filenames
eval_low_data_name.sort()
# Iterate over the list of filenames
for idx in range(len(eval_low_data_name)):
    # Split the filename into a directory path and a base name
    [_, name] = os.path.split(eval_low_data_name[idx])
    # Get the file extension (e.g., 'png')
    suffix = name[name.find('.') + 1:]
    # Get the base name of the file (without the extension)
    name = name[:name.find('.')]
    # Add the base name to the list of image names
    eval_img_name.append(name)
    # Load the image using the 'load_images' function
    eval_low_im = load_images(eval_low_data_name[idx])
    # Add the image to the list of low-resolution images
    eval_low_data.append(eval_low_im)
    # Print the shape of the image
    print(eval_low_im.shape)
# To get better results, the illumination adjustment ratio is computed based on the decom_i_high, so we also need the high data.
# Define an empty list to store the high-resolution images
eval_high_data = []
# Get a list of the filenames of the high-resolution images in the 'high' subdirectory of the test directory
eval_high_data_name = glob(args.test_dir + '/high/*.png')
# Sort the list of filenames
eval_high_data_name.sort()
# Iterate over the list of filenames
for idx in range(len(eval_high_data_name)):
    # Load the image using the 'load_images' function
    eval_high_im = load_images(eval_high_data_name[idx])
    # Add the image to the list of high-resolution images
    eval_high_data.append(eval_high_im)

# Set the directory where the enhanced images will be saved
sample_dir = args.save_dir +'/LOLdataset/'
# Create the directory if it does not already exist
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

# Print a message
print("Start evalating!")
# Record the start time
start_time = time.time()
# Iterate over the list of low-resolution images
for idx in range(len(eval_low_data)):
    print(idx)
    # Get the name of the current image
    name = eval_img_name[idx]
    print('Evaluate image %s'%name)
    # Get the current low-resolution image
    input_low = eval_low_data[idx]
    # Add a batch dimension to the low-resolution image
    input_low_eval = np.expand_dims(input_low, axis=0)
    # Get the corresponding high-resolution image
    input_high = eval_high_data[idx]
    # Add a batch dimension to the high-resolution image
    input_high_eval = np.expand_dims(input_high, axis=0)
    # Get the shape of the low-resolution image
    h, w, _ = input_low.shape

    # Decompose the low-resolution image into a reflectance and an illumination component using the DecomNet model
    decom_r_low, decom_i_low = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_low_eval})
    # Decompose the high-resolution image into a reflectance and an illumination component using the DecomNet model
    decom_r_high, decom_i_high = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_high_eval})

    # Denoise the low-resolution reflectance component using the Denoise_Net model
    restoration_r = sess.run(output_r, feed_dict={input_low_r: decom_r_low, input_low_i: decom_i_low})

    # Calculate the ratio of the mean illumination of the high-resolution image to the mean illumination of the low-resolution image
    ratio = np.mean(((decom_i_high))/(decom_i_low+0.0001))
    # Calculate the ratio of the mean reflectance of the high-resolution image to the mean reflectance of the denoised low-resolution image
    ratio2 = np.mean(((decom_r_high))/(restoration_r+0.0001))
    # If the ratio of reflectances is less than 1.1, set the illumination ratio to the illumination ratio calculated above
    if ratio2<1.1:
        i_low_data_ratio = np.ones([h, w])*(ratio)
        # Otherwise, set the illumination ratio to the sum of the illumination ratio and the reflectance ratio
    else:
        i_low_data_ratio = np.ones([h, w])*(ratio+ratio2)

    # Add a channel dimension to the illumination ratio
    i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis =2)
    # Add a batch dimension to the illumination ratio
    i_low_ratio_expand2 = np.expand_dims(i_low_ratio_expand, axis=0)

    # Adjust the low-resolution illumination component using the Illumination_adjust_net model
    adjust_i = sess.run(output_i, feed_dict={input_low_i: decom_i_low, input_low_i_ratio: i_low_ratio_expand2})
    # Multiply the denoised low-resolution reflectance component by the adjusted low-resolution illumination component to get the enhanced image
    fusion = restoration_r*adjust_i

    # Save the enhanced image to the specified directory
    save_images(os.path.join(sample_dir, '%s_kindle_v2.png' % (name)), fusion)
    #save_images(os.path.join(sample_dir, '%s_decom_i_low.png' % (name)), decom_i_low)
    #save_images(os.path.join(sample_dir, '%s_adjust_i_%f.png' % (name, (ratio+ratio2)) ), adjust_i)
    #save_images(os.path.join(sample_dir, '%s_denoise_r.png' % (name)), restoration_r)
