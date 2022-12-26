# coding: utf-8
from __future__ import print_function
import os, time, random
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import *
from model import *
from glob import glob
import argparse
# create an argument parser to parse command-line arguments
parser = argparse.ArgumentParser(description='')
# add an argument to specify the batch size
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='number of samples in one batch')
# add an argument to specify the patch size
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
# add an argument to specify the directory for training inputs
parser.add_argument('--train_data_dir', dest='train_data_dir', default='./LOLdataset/our485', help='directory for training inputs')
# add an argument to specify the directory for decomnet training results
parser.add_argument('--train_result_dir', dest='train_result_dir', default='./decom_net_train_result/', help='directory for decomnet training results')


# parse the arguments
args = parser.parse_args()

# extract the batch size and patch size from the parsed arguments
batch_size = args.batch_size
patch_size = args.patch_size

# create a TensorFlow session
sess = tf.Session()

# create placeholders for the input low and input high tensors
input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

# call the DecomNet function on the input low and input high tensors and assign the returned values to R_low, I_low, R_high, and I_high
[R_low, I_low] = DecomNet(input_low)
[R_high, I_high] = DecomNet(input_high)

# concatenate I_low and I_high along the fourth dimension with a size of 3, and assign the resulting tensors to I_low_3 and I_high_3
I_low_3 = tf.concat([I_low, I_low, I_low], axis=3)
I_high_3 = tf.concat([I_high, I_high, I_high], axis=3)

#network output
output_R_low = R_low
output_R_high = R_high
output_I_low = I_low_3
output_I_high = I_high_3

# define loss

def mutual_i_loss(input_I_low, input_I_high):
    # calculate the gradient of input_I_low along the x axis
    low_gradient_x = gradient(input_I_low, "x")
    # calculate the gradient of input_I_high along the x axis
    high_gradient_x = gradient(input_I_high, "x")
    # calculate the loss between low_gradient_x and high_gradient_x using an exponential function
    x_loss = (low_gradient_x + high_gradient_x)* tf.exp(-10*(low_gradient_x+high_gradient_x))
    # calculate the gradient of input_I_low along the y axis
    low_gradient_y = gradient(input_I_low, "y")
    # calculate the gradient of input_I_high along the y axis
    high_gradient_y = gradient(input_I_high, "y")
    # calculate the loss between low_gradient_y and high_gradient_y using an exponential function
    y_loss = (low_gradient_y + high_gradient_y) * tf.exp(-10*(low_gradient_y+high_gradient_y))
    # calculate the mean of x_loss and y_loss as the mutual loss
    mutual_loss = tf.reduce_mean( x_loss + y_loss) 
    return mutual_loss

def mutual_i_input_loss(input_I_low, input_im):
    # convert input_im to grayscale
    input_gray = tf.image.rgb_to_grayscale(input_im)
    # calculate the gradient of input_I_low along the x axis
    low_gradient_x = gradient(input_I_low, "x")
    # calculate the gradient of input_gray along the x axis
    input_gradient_x = gradient(input_gray, "x")
    # calculate the loss between low_gradient_x and input_gradient_x using the absolute value of their ratio
    x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01)))
    # calculate the gradient of input_I_low along the y axis
    low_gradient_y = gradient(input_I_low, "y")
    # calculate the gradient of input_gray along the y axis
    input_gradient_y = gradient(input_gray, "y")
    # calculate the loss between low_gradient_y and input_gradient_y using the absolute value of their ratio
    y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01)))
    # calculate the mean of x_loss and y_loss as the mutual loss
    mut_loss = tf.reduce_mean(x_loss + y_loss) 
    return mut_loss

# define the reconstruction loss for the low-resolution input
recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 -  input_low))
# define the reconstruction loss for the high-resolution input
recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - input_high))

# define the loss for the difference between the R values of the low- and high-resolution inputs
equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))

# define the loss for the mutual information between the I values of the low- and high-resolution inputs
i_mutual_loss = mutual_i_loss(I_low, I_high)

# define the loss for the mutual information between the I value of the high-resolution input and the high-resolution input image
i_input_mutual_loss_high = mutual_i_input_loss(I_high, input_high)
# define the loss for the mutual information between the I value of the low-resolution input and the low-resolution input image
i_input_mutual_loss_low = mutual_i_input_loss(I_low, input_low)

# define the overall loss as the sum of all the loss tensors, with various weights applied
loss_Decom = 1*recon_loss_high + 1*recon_loss_low \
               + 0.009 * equal_R_loss + 0.2*i_mutual_loss \
             + 0.15* i_input_mutual_loss_high + 0.15* i_input_mutual_loss_low

###
# define a placeholder for the learning rate
lr = tf.placeholder(tf.float32, name='learning_rate')

# create an Adam optimizer with the learning rate placeholder
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
# define the trainable variables of the neural network as those with a name containing "DecomNet"
var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
# create a training operation using the optimizer to minimize the loss
train_op_Decom = optimizer.minimize(loss_Decom, var_list = var_Decom)
# initialize all the variables
sess.run(tf.global_variables_initializer())

# create a saver to save the variables of the neural network during training
saver_Decom = tf.train.Saver(var_list = var_Decom)
# print a message indicating that the model has been initialized successfully
print("[*] Initialize model successfully...")

#load data
###train_data
# load the low-resolution and high-resolution training images
train_low_data = []
train_high_data = []
train_low_data_names = glob(args.train_data_dir + '/low/*.png')
train_low_data_names.sort()
train_high_data_names = glob(args.train_data_dir + '/high/*.png') 
train_high_data_names.sort()
# check that the number of low-resolution and high-resolution images is the same
assert len(train_low_data_names) == len(train_high_data_names)
# print the number of training data
print('[*] Number of training data: %d' % len(train_low_data_names))
# load the low-resolution and high-resolution images into the train_low_data and
for idx in range(len(train_low_data_names)):
    low_im = load_images(train_low_data_names[idx])
    train_low_data.append(low_im)
    high_im = load_images(train_high_data_names[idx])
    train_high_data.append(high_im)
###eval_data
# load the low-resolution and high-resolution evaluation images
eval_low_data = []
eval_high_data = []
eval_low_data_name = glob('./LOLdataset/eval15/low/*.png')
eval_low_data_name.sort()
eval_high_data_name = glob('./LOLdataset/eval15/high/*.png*')
eval_high_data_name.sort()
# load the low-resolution and high-resolution evaluation images into the eval_low_data and eval_high_data lists, respectively
for idx in range(len(eval_low_data_name)):
    eval_low_im = load_images(eval_low_data_name[idx])
    eval_low_data.append(eval_low_im)
    eval_high_im = load_images(eval_high_data_name[idx])
    eval_high_data.append(eval_high_im)


epoch = 2500
learning_rate = 0.0001
# Set the directory for saving samples during training
sample_dir = args.train_result_dir
# If the directory does not exist, create it
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)
# Set the frequency for evaluating the model using the evaluation data
eval_every_epoch = 500
# Set the phase of training
train_phase = 'decomposition'
# Calculate the number of batches
numBatch = len(train_low_data) // int(batch_size)
# Set the variables for training the neural network
train_op = train_op_Decom
train_loss = loss_Decom
saver = saver_Decom

# Set the directory for saving checkpoints of the trained model
checkpoint_dir = './checkpoint/decom_net_retrain/'
# If the directory does not exist, create it
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    # If a checkpoint was found, restore the model and set the start_step and start_epoch variables
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    # If no checkpoint was found, print a message indicating that no pretrained model was found
    print('No decomnet pretrained model!')

# Set the start step and start epoch variables
start_step = 0
start_epoch = 0
iter_num = 0

# Print a message indicating the start of training
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
# Start the timer
start_time = time.time()
# Set the initial image index to 0
image_id = 0
# Iterate through the number of epochs
for epoch in range(start_epoch, epoch):
    # Iterate through the number of batches
    for batch_id in range(start_step, numBatch):
        # Initialize arrays for low-resolution and high-resolution batch inputs
        batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        # Iterate through the patches in the current batch
        for patch_id in range(batch_size):
            # Get the shape of the current low-resolution image
            h, w, _ = train_low_data[image_id].shape
            # Select a random patch from the image
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            # Select a random data augmentation mode
            rand_mode = random.randint(0, 7)
            # Apply data augmentation to the selected patch and store it in the batch input array for low-resolution images
            batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
            # Apply data augmentation to the corresponding patch in the high-resolution image and store it in the batch input array for high-resolution images
            batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
            # Increment the image index, wrapping around to the start if necessary
            image_id = (image_id + 1) % len(train_low_data)
            # If the image index has wrapped around, shuffle the low-resolution and high-resolution images
            if image_id == 0:
                tmp = list(zip(train_low_data, train_high_data))
                random.shuffle(tmp)
                train_low_data, train_high_data  = zip(*tmp)
            # Run the training operation, computing the loss and updating the network parameters
        _, loss = sess.run([train_op, train_loss], feed_dict={input_low: batch_input_low, \
                                                              input_high: batch_input_high, \
                                                            lr: learning_rate})
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        iter_num += 1
    if (epoch + 1) % eval_every_epoch == 0:
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            result_1, result_2 = sess.run([output_R_low, output_I_low], feed_dict={input_low: input_low_eval})
            save_images(os.path.join(sample_dir, 'low_%d_%d.png' % ( idx + 1, epoch + 1)), result_1, result_2)
        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_high_data[idx], axis=0)
            result_11, result_22 = sess.run([output_R_high, output_I_high], feed_dict={input_high: input_low_eval})
            save_images(os.path.join(sample_dir, 'high_%d_%d.png' % ( idx + 1, epoch + 1)), result_11, result_22)
         
    saver.save(sess, checkpoint_dir + 'model.ckpt')

print("[*] Finish training for phase %s." % train_phase)
