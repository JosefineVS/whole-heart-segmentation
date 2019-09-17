#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:25:15 2019

@author: Josefine
"""

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
    labels_onehot = np.squeeze(np.stack((labels==0, labels==500, labels==600, labels==420, labels ==550, labels==205, labels ==820, labels==850), axis=3).astype('int32'))
    return images, labels_onehot

# Load test data
filelist_test = natural_sort(glob.glob('Data/region/validation_segments_*.npz')) # list of file names

#############################################################################
##                  Reload network and predict                         ######
#############################################################################

print("====================== LOAD AXIAL NETWORK: ===========================")
# Doing predictions with the model 
tf.reset_default_graph()      
 
new_saver = tf.train.import_meta_graph('Results/segmentation/model_axial/model.ckpt.meta')
 
prediction = np.zeros([1,256,256,9])
with tf.Session() as sess:
    new_saver.restore(sess, tf.train.latest_checkpoint('Results/segmentation/model_axial/'))
    graph = tf.get_default_graph()       
    x = graph.get_tensor_by_name("x_train:0")
    op_to_restore = graph.get_tensor_by_name("output/Reshape_1:0")
    keep_rate = graph.get_tensor_by_name("Placeholder:0")
    context = graph.get_tensor_by_name("concat_6:0")
    x_contextual = graph.get_tensor_by_name("x_train_context:0")
    for i in range(len(filelist_test)):
        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_test)))+1))         # Find renderings corresponding to the given name
        prob_maps = []
        x_test, y_test = create_data(filelist_test[i],'axial')
        for k in range(x_test.shape[0]):
            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
            y_output,out_context = sess.run([tf.nn.softmax(op_to_restore),context], feed_dict={x: x_test_image, x_contextual: prediction,keep_rate: 1.0})
            prediction[0,:,:,:] = out_context
            prob_maps.append(y_output[0,:,:,:])
        np.savez('Results/Predictions/segment/val_prob_maps_axial_{}'.format(i),prob_maps=prob_maps)                            
print("================ DONE WITH AXIAL PREDICTIONS! ==================")  
 
print("====================== LOAD SAGITTAL NETWORK: ===========================")
# Doing predictions with the model 
tf.reset_default_graph()      

new_saver = tf.train.import_meta_graph('Results/segmentation/model_sag/model.ckpt.meta')
prediction = np.zeros([1,256,256,9])
with tf.Session() as sess:
    new_saver.restore(sess, tf.train.latest_checkpoint('Results/segmentation/model_sag/'))
    graph = tf.get_default_graph()       
    x = graph.get_tensor_by_name("x_train:0")
    keep_rate = graph.get_tensor_by_name("Placeholder:0")
    op_to_restore = graph.get_tensor_by_name("output/Reshape_1:0")
    context = graph.get_tensor_by_name("concat_6:0")
    x_contextual = graph.get_tensor_by_name("x_train_context:0")
    for i in range(len(filelist_test)):
        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_test)))+1))
        # Find renderings corresponding to the given name
        prob_maps = []
        x_test, y_test = create_data(filelist_test[i],'sag')
        for k in range(x_test.shape[0]):
            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
            y_output,out_context = sess.run([tf.nn.softmax(op_to_restore),context], feed_dict={x: x_test_image, x_contextual: prediction,keep_rate: 1.0})
            prediction[0,:,:,:] = out_context
            prob_maps.append(y_output[0,:,:,:])
        np.savez('Results/Predictions/segment/val_prob_maps_sag_{}'.format(i),prob_maps=prob_maps)                            
print("================ DONE WITH SAGITTAL PREDICTIONS! ==================")  

print("====================== LOAD CORONAL NETWORK: ===========================")
# Doing predictions with the model 
tf.reset_default_graph()      

new_saver = tf.train.import_meta_graph('Results/segmentation/model_cor/model.ckpt.meta')
prediction = np.zeros([1,256,256,9])
with tf.Session() as sess:
    new_saver.restore(sess, tf.train.latest_checkpoint('Results/segmentation/model_cor/'))
    graph = tf.get_default_graph()       
    x = graph.get_tensor_by_name("x_train:0")
    keep_rate = graph.get_tensor_by_name("Placeholder:0")
    op_to_restore = graph.get_tensor_by_name("output/Reshape_1:0")
    context = graph.get_tensor_by_name("concat_6:0")
    x_contextual = graph.get_tensor_by_name("x_train_context:0")
    for i in range(len(filelist_test)):
        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_test)))+1))
        # Find renderings corresponding to the given name
        prob_maps = []
        x_test, y_test = create_data(filelist_test[i],'cor')
        for k in range(x_test.shape[0]):
            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
            y_output,out_context = sess.run([tf.nn.softmax(op_to_restore),context], feed_dict={x: x_test_image, x_contextual: prediction,keep_rate: 1.0})
            prediction[0,:,:,:] = out_context
            prob_maps.append(y_output[0,:,:,:])
        np.savez('Results/Predictions/segment/val_prob_maps_cor_{}'.format(i),prob_maps=prob_maps)                            
print("================ DONE WITH CORONAL PREDICTONS! ==================")  

