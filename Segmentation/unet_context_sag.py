#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:33:11 2018

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
from skimage.transform import resize

# Define parameters:
lr          = 1e-5    # learning-rate
nEpochs     = 30         # Number of epochs

# Other network specific parameters
n_classes = 8
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

imgDim = 256
labelDim = 256
######################################################################
##                                                                  ##
##                   Setting up the network                         ##
##                                                                  ##
######################################################################

tf.reset_default_graph()

#Define placeholder for input and output
x = tf.placeholder(tf.float32,[None,imgDim,imgDim,1],name = 'x_train') #input (572+572+1 image)
x_contextual = tf.placeholder(tf.float32,[None,imgDim,imgDim,9],name = 'x_train_context') #input (572+572+1 image)
y = tf.placeholder(tf.float32,[None,labelDim,labelDim,n_classes],name='y_train') #Output (388x388x2 labels)
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

#def dice_coef(y, output): #making the loss function smooth
#    y_true_f = tf.contrib.layers.flatten(tf.argmax(y,axis=-1))
#    y_pred_f = tf.contrib.layers.flatten(tf.argmax(output,axis=-1))
#    intersection = tf.reduce_sum(y_true_f * y_pred_f)
#    return (2 * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f))

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

def dropout(input1,drop_rate):
    input_shape = input1.get_shape().as_list()
    noise_shape = tf.constant(value=[1, 1, 1, input_shape[3]])
    drop = tf.nn.dropout(input1, keep_prob=drop_rate, noise_shape=noise_shape)
    return drop

def crop2d(inputs,dim):
    crop = tf.image.resize_image_with_crop_or_pad(inputs,dim,dim)
    return crop

def concat(input1,input2,axis):
    combined = tf.concat([input1,input2],axis)
    return combined

def transpose(inputs,filters, kernel, stride, pad, name):
    with tf.name_scope(name):
        trans = tf.layers.conv2d_transpose(inputs,filters, kernel_size=[kernel,kernel],strides=[stride,stride],padding=pad,kernel_initializer=tf.contrib.layers.xavier_initializer())
        return trans
    
######################################################################
##                             Data                                 ##
###################################################################### 
    
def create_data(filename_img,direction):
    images = []
    file = np.load(filename_img)
    a = file['images']
    # Reshape:
    im = resize(a,(labelDim,labelDim,labelDim),order=0)
    if direction == 'axial':
        for i in range(im.shape[0]):
            images.append((im[i,:,:]))
    if direction == 'sag':
        for i in range(im.shape[1]):
            images.append((im[:,i,:]))
    if direction == 'cor':
        for i in range(im.shape[2]):
            images.append((im[:,:,i]))    
    images = np.asarray(images)
    images = images.reshape(-1, imgDim,imgDim,1)

    # Label creation
    labels = []
    b = file['labels']        
    lab = resize(b,(labelDim,labelDim,labelDim),order=0)
    if direction == 'axial':
        for i in range(lab.shape[0]):
            labels.append((lab[i,:,:]))
    if direction == 'sag':
        for i in range(lab.shape[1]):
            labels.append((lab[:,i,:]))
    if direction == 'cor':
        for i in range(lab.shape[2]):
            labels.append((lab[:,:,i]))            
    labels = np.asarray(labels)
    labels_onehot = np.stack((labels==0, labels==500, labels==600, labels==420, labels ==550, labels==205, labels ==820, labels==850), axis=3)

    return images, labels_onehot


###############################################################################
##                            Setup of network                               ##
###############################################################################

# -------------------------- Contracting path ---------------------------------
conv1a = conv2d(x,filters=64,kernel=3,stride=1,pad='same',name = 'conv1a')
conv1a.get_shape()
conv1b = conv2d(conv1a,filters=64,kernel=3,stride=1,pad='same',name = 'conv1b')
conv1b.get_shape()
#drop1 = tf.nn.dropout(conv1b, keep_prob=drop_rate) 
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

# -------------------------- Contextual input path ----------------------------

conv1a_2 = conv2d(x_contextual,filters=64,kernel=3,stride=1,pad='same',name = 'conv1a2')
conv1b_2 = conv2d(conv1a_2,filters=64,kernel=3,stride=1,pad='same',name = 'conv1b2')
#drop1_2 = tf.nn.dropout(conv1b_2, keep_prob=drop_rate) 
pool1_2 = max_pool(conv1b_2,n=2,stride=2,pad='SAME')

conv2a_2 = conv2d(pool1_2,filters=128,kernel=3,stride=1,pad='same',name = 'conv2a2')
conv2b_2 = conv2d(conv2a_2,filters=128,kernel=3,stride=1,pad='same',name = 'conv2b2')
drop2_2 = dropout(conv2b_2, drop_rate) 
pool2_2 = max_pool(drop2_2,n=2,stride=2,pad='SAME')

conv3a_2 = conv2d(pool2_2,filters=256,kernel=3,stride=1,pad='same',name = 'conv3a2')
conv3b_2 = conv2d(conv3a_2,filters=256,kernel=3,stride=1,pad='same',name = 'conv3b2')
drop3_2 = dropout(conv3b_2, drop_rate)  
pool3_2 = max_pool(drop3_2,n=2,stride=2,pad='SAME')

conv4a_2 = conv2d(pool3_2,filters=512,kernel=3,stride=1,pad='same',name = 'conv4a2')
conv4b_2 = conv2d(conv4a_2,filters=512,kernel=3,stride=1,pad='same',name = 'conv4b2')
drop4_2 = dropout(conv4b_2, drop_rate) 
pool4_2 = max_pool(drop4_2,n=2,stride=2,pad='SAME')

# ---------------------------- Expansive path ---------------------------------
combx = concat(pool4,pool4_2,axis=3)
conv5a = conv2d(combx,filters=1024,kernel=3,stride=1,pad='same',name = 'conv5a')
conv5a.get_shape()
conv5b = conv2d(conv5a,filters=1024,kernel=3,stride=1,pad='same',name = 'conv5b')
conv5b.get_shape()
drop5 = dropout(conv5b, drop_rate) 
drop5.get_shape()
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
#drop9 = tf.nn.dropout(conv9b, keep_prob=drop_rate) 
#drop9.get_shape()
up9a = transpose(conv9b,filters=64,kernel=2,stride=2,pad='same',name='up7a')
up9a.get_shape()
up9b = concat(up9a,conv1b,axis=3)
up9b.get_shape()

conv10a = conv2d(up9b,filters=64,kernel=3,stride=1,pad='same',name = 'conv7a')
conv10a.get_shape()
conv10b = conv2d(conv10a,filters=64,kernel=3,stride=1,pad='same',name = 'conv7b')
conv10b.get_shape()

output = tf.layers.conv2d(conv10b, n_classes, 1, (1,1),padding ='same',activation=tf.nn.softmax, kernel_initializer=tf.contrib.layers.xavier_initializer(), name = 'output')
output.get_shape()

######################################################################
##                                                                  ##
##                            Loading data                          ##
##                                                                  ##
######################################################################

filelist_train = natural_sort(glob.glob('WHS/Data/train_segments_*.npz')) # list of file names
x_train = {}
y_train = {}
keys = range(len(filelist_train))
for i in keys:
    x_train[i] = np.zeros([imgDim,imgDim,imgDim,1])
    y_train[i] = np.zeros([imgDim,imgDim,imgDim,8])

for i in range(len(filelist_train)):
    img, lab = create_data(filelist_train[i],'sag')
    x_train[i] = img
    y_train[i] = lab    

#filelist_val = natural_sort(glob.glob('WHS/Data/validation_segments_*.npz')) # list of file names
#x_val = {}
#y_val = {}
#keys = range(len(filelist_val))
#for i in keys:
#    x_val[i] = np.zeros([imgDim,imgDim,imgDim,1])
#    y_val[i] = np.zeros([imgDim,imgDim,imgDim,8])
#
#for i in range(len(filelist_val)):
#    img, lab = create_data(filelist_val[i],'sag')
#    x_val[i] = img
#    y_val[i] = lab    
#        
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
loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y_true = y, y_pred = output))
correct_prediction = tf.equal(tf.argmax(output, axis=-1), tf.argmax(y, axis=-1))

# averaging the one-hot encoded vector
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#dice = dice_coef(y, output,smooth=1)

# Create contextual output:
pred = tf.argmax(tf.nn.softmax(output[0,:,:,:]),axis=-1)
predict = tf.one_hot(pred,8)
context = tf.concat([x[0,:,:,:],predict],axis=-1)

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
valid_loss, valid_accuracy = [], []
train_loss, train_accuracy = [], []

c = np.zeros([imgDim+1,imgDim,imgDim,9])
predictions = {}
keys = range(len(filelist_train))
for i in keys:
    predictions[i] = c

#predictions_val = {}
#keys = range(len(filelist_val))
#for i in keys:
#    predictions_val[i] = c

index_volumeID = np.repeat(range(len(x_train)),imgDim)
index_imageID = np.tile(range(imgDim),len(x_train))
index_comb = np.vstack((index_volumeID,index_imageID)).T

index_shuffle = shuffle(index_comb)
with tf.Session() as sess:
    # Initialize
    t_start = time()

    sess.run(init)    
    
    # Trainingsloop
    for epoch in range(nEpochs):
        t_epoch_start = time()
        print('========Training Epoch: ', (epoch + 1))
        iter_by_epoch = len(index_shuffle)            
        for i in range(iter_by_epoch):
            t_iter_start = time()
            x_batch = np.expand_dims(x_train[index_shuffle[i,0]][index_shuffle[i,1],:,:,:], axis=0)
            x_batch_context = np.expand_dims(predictions[index_shuffle[i,0]][index_shuffle[i,1],:,:,:], axis=0)
            y_batch = np.expand_dims(y_train[index_shuffle[i,0]][index_shuffle[i,1],:,:,:], axis=0)
            _,_loss,_acc,pred_out = sess.run([train_adam, loss, accuracy,context], feed_dict={x: x_batch, x_contextual: x_batch_context, y: y_batch, drop_rate: 0.5})   
            predictions[index_shuffle[i,0]][index_shuffle[i,1]+1,:,:,:] = pred_out
            train_loss.append(_loss)
            train_accuracy.append(_acc)

#            # Validation-step:
#            if i==np.max(range(iter_by_epoch)):
#                for n in range(len(x_val)):
#                    for m in range(imgDim):
#                        x_batch_val = np.expand_dims(x_val[n][m,:,:,:], axis=0)
#                        y_batch_val = np.expand_dims(y_val[n][m,:,:,:], axis=0)
#                        x_context_val = np.expand_dims(predictions_val[n][m,:,:,:], axis=0)
#                        acc_val, loss_val,out_context = sess.run([accuracy,loss,context], feed_dict={x: x_batch_val, x_contextual: x_context_val, y: y_batch_val, drop_rate: 1.0})
#                        predictions_val[n][m+1,:,:,:] = pred_out
#                        valid_loss.append(loss_val)
#                        valid_accuracy.append(acc_val)                        
#       
        t_epoch_finish = time() 
        print("Epoch:", (epoch + 1), '  avg_loss= ', "{:.9f}".format(np.mean(train_loss)), '  avg_acc= ', "{:.9f}".format(np.mean(train_accuracy)),' time_epoch=', str(t_epoch_finish-t_epoch_start))
#        print("Validation:", (epoch + 1), '  avg_loss= ', "{:.9f}".format(np.mean(valid_loss)), '  avg_acc= ', "{:.9f}".format(np.mean(valid_accuracy)))

    t_end = time()

    saver.save(sess,"WHS/Results/segmentation/model_sag/model.ckpt")
    np.save('WHS/Results/train_hist/segmentation/train_loss_sag',train_loss)
    np.save('WHS/Results/train_hist/segmentation/train_acc_sag',train_accuracy)
#    np.save('WHS/Results/train_hist/segmentation/valid_loss_sag',valid_loss)
#    np.save('WHS/Results/train_hist/segmentation/valid_acc_sag',valid_accuracy)
    print('Training Done! Total time:' + str(t_end - t_start))#!/usr/bin/env python3
