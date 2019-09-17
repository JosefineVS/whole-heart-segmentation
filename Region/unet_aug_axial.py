#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:28:59 2018

@author: Josefine
"""

## Import libraries
import numpy as np 
import tensorflow as tf
import re
import glob
import keras
from time import time
from sklearn.utils import shuffle
import nibabel as nib
from skimage.transform import resize

# Define parameters:
lr          = 1e-5    # learning-rate (or starting LR if it is decreasing)
nEpochs     = 50         # Number of epochs
batch_size  = 1
valid_size  = 1

# Other network specific parameters
n_classes = 2
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

imgDim = 128
######################################################################
##                                                                  ##
##                   Setting up the network                         ##
##                                                                  ##
######################################################################

tf.reset_default_graph()

#Define placeholder for input and output
x = tf.placeholder(tf.float32,[None,imgDim,imgDim,1],name = 'x_train') #input (572+572+1 image)
y = tf.placeholder(tf.float32,[None,imgDim,imgDim,n_classes],name='y_train') #Output (388x388x2 labels)
drop_rate = tf.placeholder(tf.float32, shape=())

######################################################################
##                                                                  ##
##                   Metrics and functions                          ##
##                                                                  ##
######################################################################

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def dice_coef(y_true, y_pred): #making the loss function smooth
    y_true_f = tf.contrib.layers.flatten(tf.argmax(y,axis=-1))
    y_pred_f = tf.contrib.layers.flatten(tf.argmax(output,axis=-1))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2 * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f))

######################################################################
##                               Layers                             ##
######################################################################
def conv2d(inputs, filters, kernel, stride, pad, name):
    """ Creates a 2D convolution with following specs:
    Args:
        inputs:         (Tensor)            Tensor which you want to apply convolution to 
        filters:        (integer)           Number of filters in kernel
        kernel_size:    (integer)           Size of kernel
        Strides:        (integer)           Stride
        pad:            ('VALID' or 'SAME') Type of padding
        name:           (string)            Name of layer
    """
    with tf.name_scope(name):
        conv = tf.layers.conv2d(inputs, filters, kernel_size = kernel, strides = [stride,stride], padding=pad,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        return conv  

def max_pool(inputs,n,stride,pad):
    maxpool = tf.nn.max_pool(inputs, ksize=[1,n,n,1], strides=[1,stride,stride,1], padding=pad)
    return maxpool

def crop2d(inputs,dim):
    crop = tf.image.resize_image_with_crop_or_pad(inputs,dim,dim)
    return crop

def concat(input1,input2,axis):
    combined = tf.concat([input1,input2],axis)
    return combined

def dropout(input1,drop_rate):
    input_shape = input1.get_shape().as_list()
    noise_shape = tf.constant(value=[1, 1, 1, input_shape[3]])
    drop = tf.nn.dropout(input1, keep_prob=drop_rate, noise_shape=noise_shape)
    return drop

def transpose(inputs,filters, kernel, stride, pad, name):
    with tf.name_scope(name):
        trans = tf.layers.conv2d_transpose(inputs,filters, kernel_size=[kernel,kernel],strides=[stride,stride],padding=pad,kernel_initializer=tf.contrib.layers.xavier_initializer())
        return trans
    
######################################################################
##                             Data                                 ##
######################################################################

def create_data(filename_img,filename_label,direction):
    images = []
    for f in range(len(filename_img)):
        a = nib.load(filename_img[f])
        a = a.get_data()
        # Normalize:
        a2 = np.clip(a,-1000,1000)
        a3 = np.interp(a2, (a2.min(), a2.max()), (-1, +1))
        # Reshape:
        img = np.zeros([512,512,512])+np.min(a3)
        index1 = int(np.ceil((512-a.shape[2])/2))
        index2 = int(512-np.floor((512-a.shape[2])/2))
        img[:,:,index1:index2] = a3
        im = resize(img,(imgDim,imgDim,imgDim),order=0)
        if direction == 'sag':
            for i in range(im.shape[0]):
                images.append((im[i,:,:]))
        if direction == 'cor':
            for i in range(im.shape[1]):
                images.append((im[:,i,:]))
        if direction == 'axial':
            for i in range(im.shape[2]):
                images.append((im[:,:,i]))    
    images = np.asarray(images)
    images = images.reshape(-1, imgDim,imgDim,1)
    
    # Label creation
    labels = []
    for g in range(len(filename_label)):
        b = nib.load(filename_label[g])
        b = b.get_data()
        img = np.zeros([b.shape[0],b.shape[0],b.shape[0]])
        index1 = int(np.ceil((img.shape[2]-b.shape[2])/2))
        index2 = int(img.shape[2]-np.floor((img.shape[2]-b.shape[2])/2))
        img[:,:,index1:index2] = b
        lab = resize(img,(imgDim,imgDim,imgDim),order=0)
        lab[lab>1] = 1
        if direction == 'sag':
            for i in range(lab.shape[0]):
                labels.append((lab[i,:,:]))
        if direction == 'cor':
            for i in range(lab.shape[1]):
                labels.append((lab[:,i,:]))
        if direction == 'axial':
            for i in range(lab.shape[2]):
                labels.append((lab[:,:,i]))            
    labels = np.asarray(labels)
    labels_onehot = np.stack((labels==0, labels==1), axis=3).astype('int32')
    return images, labels_onehot

###############################################################################
##                            Setup of network                               ##
###############################################################################

# -------------------------- Contracting path ---------------------------------
conv1a = conv2d(x,filters=64,kernel=3,stride=1,pad='same',name = 'conv1a')
conv1a.get_shape()
conv1b = conv2d(conv1a,filters=64,kernel=3,stride=1,pad='same',name = 'conv1b')
conv1b.get_shape()
#drop1 = dropout(conv1b, drop_rate) 
#drop1.get_shape()
pool1 = max_pool(conv1b,n=2,stride=2,pad='SAME')
pool1.get_shape()

conv2a = conv2d(pool1,filters=128,kernel=3,stride=1,pad='same',name = 'conv2a')
conv2a.get_shape()
conv2b = conv2d(conv2a,filters=128,kernel=3,stride=1,pad='same',name = 'conv2b')
conv2b.get_shape()
drop2 = dropout(conv2b, drop_rate) 
drop2.get_shape()
pool2 = max_pool(drop2,n=2,stride=2,pad='SAME')
pool2.get_shape()

conv3a = conv2d(pool2,filters=256,kernel=3,stride=1,pad='same',name = 'conv3a')
conv3a.get_shape()
conv3b = conv2d(conv3a,filters=256,kernel=3,stride=1,pad='same',name = 'conv3b')
conv3b.get_shape()
drop3 = dropout(conv3b, drop_rate) 
drop3.get_shape()
pool3 = max_pool(drop3,n=2,stride=2,pad='SAME')
pool3.get_shape()

conv4a = conv2d(pool3,filters=512,kernel=3,stride=1,pad='same',name = 'conv4a')
conv4a.get_shape()
conv4b = conv2d(conv4a,filters=512,kernel=3,stride=1,pad='same',name = 'conv4b')
conv4b.get_shape()
drop4 = dropout(conv4b, drop_rate) 
drop4.get_shape()
pool4 = max_pool(drop4,n=2,stride=2,pad='SAME')
pool4.get_shape()

conv5a = conv2d(pool4,filters=1024,kernel=3,stride=1,pad='same',name = 'conv5a')
conv5a.get_shape()
conv5b = conv2d(conv5a,filters=1024,kernel=3,stride=1,pad='same',name = 'conv5b')
conv5b.get_shape()
drop5 = dropout(conv5b, drop_rate) 
drop5.get_shape()
# ---------------------------- Expansive path ---------------------------------
up6a = transpose(drop5,filters=512,kernel=2,stride=2,pad='same',name='up6a')
up6a.get_shape()
up6b = concat(up6a,conv4b,axis=3)
up6b.get_shape()

conv7a = conv2d(up6b,filters=512,kernel=3,stride=1,pad='same',name = 'conv7a')
conv7a.get_shape()
conv7b = conv2d(conv7a,filters=512,kernel=3,stride=1,pad='same',name = 'conv7b')
conv7b.get_shape()
drop7 = dropout(conv7b, drop_rate) 
drop7.get_shape()
up7a = transpose(drop7,filters=256,kernel=2,stride=2,pad='same',name='up7a')
up7a.get_shape()
up7b = concat(up7a,conv3b,axis=3)
up7b.get_shape()

conv8a = conv2d(up7b,filters=256,kernel=3,stride=1,pad='same',name = 'conv7a')
conv8a.get_shape()
conv8b = conv2d(conv8a,filters=256,kernel=3,stride=1,pad='same',name = 'conv7b')
conv8b.get_shape()
drop8 = dropout(conv8b, drop_rate) 
drop8.get_shape()
up8a = transpose(drop8,filters=128,kernel=2,stride=2,pad='same',name='up7a')
up8a.get_shape()
up8b = concat(up8a,conv2b,axis=3)
up8b.get_shape()

conv9a = conv2d(up8b,filters=128,kernel=3,stride=1,pad='same',name = 'conv7a')
conv9a.get_shape()
conv9b = conv2d(conv9a,filters=128,kernel=3,stride=1,pad='same',name = 'conv7b')
conv9b.get_shape()
#drop9 = dropout(conv9b, drop_rate) 
#drop9.get_shape()
up9a = transpose(conv9b,filters=64,kernel=2,stride=2,pad='same',name='up7a')
up9a.get_shape()
up9b = concat(up9a,conv1b,axis=3)
up9b.get_shape()

conv10a = conv2d(up9b,filters=64,kernel=3,stride=1,pad='same',name = 'conv7a')
conv10a.get_shape()
conv10b = conv2d(conv10a,filters=64,kernel=3,stride=1,pad='same',name = 'conv7b')
conv10b.get_shape()

output = tf.layers.conv2d(conv10b, 2, 1, (1,1),padding ='same',activation=tf.nn.softmax, kernel_initializer=tf.contrib.layers.xavier_initializer(), name = 'output')
output.get_shape()

######################################################################
##                                                                  ##
##                            Loading data                          ##
##                                                                  ##
######################################################################

filelist_train = natural_sort(glob.glob('WHS/Augment_data/*_image.nii')) # list of file names
filelist_train_label = natural_sort(glob.glob('WHS/Augment_data/*_label.nii')) # list of file names
x_data, y_data = create_data(filelist_train,filelist_train_label,'axial')

#filelist_val = natural_sort(glob.glob('WHS/validation/*_image.nii.gz')) # list of file names
#filelist_val_label = natural_sort(glob.glob('WHS/validation/*_label.nii.gz')) # list of file names
#x_val, y_val = create_data(filelist_val,filelist_val_label,'axial')

######################################################################
##                                                                  ##
##                   Defining the training                          ##
##                                                                  ##
######################################################################

# Training-steps (honestly I have no idea what it does...)
global_step = tf.Variable(0,trainable=False)

###############################################################################
##                               Loss                                        ##
###############################################################################
# Compare the output of the network (output: tensor) with the ground truth (y: tensor/placeholder)
# In this case we use sigmoid cross entropu losss with logits
loss = tf.reduce_mean(keras.losses.binary_crossentropy(y_true = y, y_pred = output))

# accuracy and dice 
correct_prediction = tf.equal(tf.argmax(output, axis=-1), tf.argmax(y, axis=-1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
dice = dice_coef(tf.argmax(y,axis=-1), tf.argmax(output,axis=-1))

###############################################################################
##                               Optimizer                                   ##
###############################################################################
opt = tf.train.AdamOptimizer(lr,beta1,beta2,epsilon)

###############################################################################
##                               Minimizer                                   ##
###############################################################################
train_adam = opt.minimize(loss, global_step)

###############################################################################
##                               Initializer                                 ##
###############################################################################
# Initializes all variables in the graph
init = tf.global_variables_initializer()

######################################################################
##                                                                  ##
##                   Start training                                 ## 
##                                                                  ##
######################################################################
# Initialize saving of the network parameters:
saver = tf.train.Saver()

######################## Start training Session ###########################
start_time = time()
#valid_loss, valid_accuracy, valid_dice = [], [], []
train_loss, train_accuracy, train_dice = [], [], []

index_train = shuffle(range(x_data.shape[0]))
#valid_size = int(np.floor(len(index1)*0.1))
#index_train = index1[valid_size:]
#index_valid = index1[:valid_size]
with tf.Session() as sess:
    t_start = time()
    # Initialize
    sess.run(init)    
  
    # Trainingsloop
    for epoch in range(nEpochs):
        t_epoch_start = time()
        print('========Training Epoch: ', (epoch + 1))
        iter_by_epoch = len(index_train)
        index_train_shuffle = shuffle(index_train)
        for i in range(iter_by_epoch):
            t_iter_start = time()
            x_batch = np.expand_dims(x_data[index_train_shuffle[i],:,:,:], axis=0)
            y_batch = np.expand_dims(y_data[index_train_shuffle[i],:,:,:], axis=0)
            _,_loss,_acc,_dice= sess.run([train_adam, loss, accuracy,dice], feed_dict = {x: x_batch, y: y_batch, drop_rate: 0.5})    
            
            train_loss.append(_loss)
            train_accuracy.append(_acc)
            train_dice.append(_dice)
#                 
#            # Validation-step:
#            if i==np.max(range(iter_by_epoch)):
#                valid_range = x_val.shape[0]
#                for m in range(valid_range):
#                    x_batch_val = np.expand_dims(x_val[m,:,:,:], axis=0)
#                    y_batch_val = np.expand_dims(y_val[m,:,:,:], axis=0)
#                    _loss_valid,_acc_valid,_dice_valid, = sess.run([loss,accuracy,dice], feed_dict= {x: x_batch_val,y: y_batch_val, drop_rate: 1.0})
#                    valid_loss.append(_loss_valid)
#                    valid_accuracy.append(_acc_valid)
#                    valid_dice.append(_dice_valid)

        t_epoch_finish = time() 
        print("Epoch:", (epoch + 1), '  avg_loss= ', "{:.9f}".format(np.mean(train_loss)), 'avg_acc= ', "{:.9f}".format(np.mean(train_accuracy)),'avg_dice= ', "{:.9f}".format(np.mean(train_dice)),' time_epoch=', str(t_epoch_finish-t_epoch_start))
#        print("Validation:", (epoch + 1), '  avg_loss= ', "{:.9f}".format(np.mean(valid_loss)), '  avg_acc= ', "{:.9f}".format(np.mean(valid_accuracy)),'avg_dice= ', "{:.9f}".format(np.mean(valid_dice)))

    t_end = time()
    # Save the model in the end
    saver.save(sess,"WHS/Results/region/model_axial/model.ckpt")
    np.save('WHS/Results/train_hist/region/train_loss_axial',train_loss)
    np.save('WHS/Results/train_hist/region/train_acc_axial',train_accuracy)
#    np.save('WHS/Results/train_hist/region/valid_loss_axial',valid_loss)
#    np.save('WHS/Results/train_hist/region/valid_acc_axial',valid_accuracy)
    print('Training Done! Total time:' + str(t_end - t_start))