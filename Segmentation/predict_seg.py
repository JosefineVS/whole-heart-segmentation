#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:47:22 2018

@author: Josefine
"""

import tensorflow as tf
import numpy as np
import glob
import re
from skimage.transform import resize

imgDim = 256
labelDim = 256

##############################################################################
###                              Data functions                         ######
##############################################################################
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def create_data(filename_img,direction):
    images = []
    file = np.load(filename_img)
    a = file['images']
    # Normalize:
    #a2 = np.clip(a,-1000,1000)
    #a3 = np.interp(a2, (a2.min(), a2.max()), (-1, +1))
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
    return images

# Load test data
filelist_test = natural_sort(glob.glob('WHS/Data/test_segments_*.npz')) # list of file names

#############################################################################
##                  Reload network and predict                         ######
#############################################################################
#
## =============================================================================
print("====================== LOAD AXIAL NETWORK: ===========================")
# Doing predictions with the model 
tf.reset_default_graph()      

new_saver = tf.train.import_meta_graph('WHS/Results/segmentation/model_axial/model.ckpt.meta')
 
prediction = np.zeros([1,256,256,9])
with tf.Session() as sess:
    new_saver.restore(sess, tf.train.latest_checkpoint('WHS/Results/segmentation/model_axial/'))
    graph = tf.get_default_graph()       
    x = graph.get_tensor_by_name("x_train:0")
    op_to_restore = graph.get_tensor_by_name("output/Softmax:0")
    keep_rate = graph.get_tensor_by_name("Placeholder:0")
    context = graph.get_tensor_by_name("concat_5:0")
    x_contextual = graph.get_tensor_by_name("x_train_context:0")
    for i in range(30,len(filelist_test)):
        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_test)))+1))
        # Find renderings corresponding to the given name
        prob_maps = []
        x_test = create_data(filelist_test[i],'axial')
        for k in range(x_test.shape[0]):
            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
            y_output,out_context = sess.run([tf.nn.softmax(op_to_restore),context], feed_dict={x: x_test_image, x_contextual: prediction,keep_rate: 1.0})
            prediction[0,:,:,:] = out_context
            prob_maps.append(y_output[0,:,:,:])
        np.savez('WHS/Results/Predictions/segment/train_prob_maps_axial_{}'.format(i),prob_maps=prob_maps)                            
print("================ DONE WITH AXIAL PREDICTIONS! ==================")  
#
# =============================================================================
#print("====================== LOAD SAGITTAL NETWORK: ===========================")
## Doing predictions with the model 
#tf.reset_default_graph()      
#
#new_saver = tf.train.import_meta_graph('WHS/Results/segmentation/model_sag/model.ckpt.meta')
#prediction = np.zeros([1,256,256,9])
#with tf.Session() as sess:
#    new_saver.restore(sess, tf.train.latest_checkpoint('WHS/Results/segmentation/model_sag/'))
#    graph = tf.get_default_graph()       
#    x = graph.get_tensor_by_name("x_train:0")
#    keep_rate = graph.get_tensor_by_name("Placeholder:0")
#    op_to_restore = graph.get_tensor_by_name("output/Softmax:0")
#    context = graph.get_tensor_by_name("concat_5:0")
#    x_contextual = graph.get_tensor_by_name("x_train_context:0")
#    for i in range(30,len(filelist_test)):
#        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_test)))+1))
#        # Find renderings corresponding to the given name
#        prob_maps = []
#        x_test = create_data(filelist_test[i],'sag')
#        for k in range(x_test.shape[0]):
#            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
#            y_output,out_context = sess.run([tf.nn.softmax(op_to_restore),context], feed_dict={x: x_test_image, x_contextual: prediction,keep_rate: 1.0})
#            prediction[0,:,:,:] = out_context
#            prob_maps.append(y_output[0,:,:,:])
#        np.savez('WHS/Results/Predictions/segment/train_prob_maps_sag_{}'.format(i),prob_maps=prob_maps)                            
#print("================ DONE WITH SAGITTAL PREDICTIONS! ==================")  
##
#print("====================== LOAD CORONAL NETWORK: ===========================")
## Doing predictions with the model 
#tf.reset_default_graph()      
#
#new_saver = tf.train.import_meta_graph('WHS/Results/segmentation/model_cor/model.ckpt.meta')
#prediction = np.zeros([1,256,256,9])
#with tf.Session() as sess:
#    new_saver.restore(sess, tf.train.latest_checkpoint('WHS/Results/segmentation/model_cor/'))
#    graph = tf.get_default_graph()       
#    x = graph.get_tensor_by_name("x_train:0")
#    keep_rate = graph.get_tensor_by_name("Placeholder:0")
#    op_to_restore = graph.get_tensor_by_name("output/Softmax:0")
#    context = graph.get_tensor_by_name("concat_5:0")
#    x_contextual = graph.get_tensor_by_name("x_train_context:0")
#    for i in range(30,len(filelist_test)):
#        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_test)))+1))
#        # Find renderings corresponding to the given name
#        prob_maps = []
#        x_test = create_data(filelist_test[i],'cor')
#        for k in range(x_test.shape[0]):
#            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
#            y_output,out_context = sess.run([tf.nn.softmax(op_to_restore),context], feed_dict={x: x_test_image, x_contextual: prediction,keep_rate: 1.0})
#            prediction[0,:,:,:] = out_context
#            prob_maps.append(y_output[0,:,:,:])
#        np.savez('WHS/Results/Predictions/segment/train_prob_maps_cor_{}'.format(i),prob_maps=prob_maps)                            
#print("================ DONE WITH CORONAL PREDICTONS! ==================")  
#
