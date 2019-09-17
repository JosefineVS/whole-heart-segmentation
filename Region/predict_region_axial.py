#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:05:57 2018

@author: Josefine
"""

import tensorflow as tf
import numpy as np
import nibabel as nib
import glob
import re
from skimage.transform import resize

imgDim = 128

##############################################################################
###                              Data functions                         ######
##############################################################################
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def create_image(filename_img,direction):
    images = []
    a = nib.load(filename_img)
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
    return images

# Load test data
filelist_test = natural_sort(glob.glob('WHS/ct_train_test/ct_test/*_image.nii.gz')) # list of file names

# Load train data for segmentation network
filelist_train = natural_sort(glob.glob('WHS/Augment_data/*_image.nii')) # list of file names

##############################################################################
###                  Reload network and predict                         ######
##############################################################################
print("====================== LOAD AXIAL NETWORK: ===========================")

# Doing predictions with the model 
tf.reset_default_graph()      

new_saver = tf.train.import_meta_graph('WHS/Results/region/model_axial/model.ckpt.meta')

with tf.Session() as sess:
    new_saver.restore(sess, tf.train.latest_checkpoint('WHS/Results/region/model_axial/'))
    graph = tf.get_default_graph()       
    x = graph.get_tensor_by_name("x_train:0")
    op_to_restore = graph.get_tensor_by_name("output/Softmax:0") #ME

#    for i in range(len(filelist_test)):
#        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_test)))+1))
#        # Find renderings corresponding to the given name
#        prob_maps = []
#        x_test = create_image(filelist_test[i],'axial')
#        for k in range(x_test.shape[0]):
#            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
#            y_output = sess.run(tf.nn.softmax(op_to_restore), feed_dict={x: x_test_image,'Placeholder:0':1.0})
#            prob_maps.append(y_output[0,:,:,:])
#        np.savez('WHS/Results/Predictions/region/test_prob_maps_axial_{}'.format(i),prob_maps=prob_maps)                            
#    print("================ DONE WITH TEST PREDICTIONS! ==================")  

    for i in range(30,len(filelist_train)):
        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_train)))+1))
        # Find renderings corresponding to the given name
        prob_maps = []
        x_test = create_image(filelist_train[i],'axial')
        for k in range(x_test.shape[0]):
            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
            y_output = sess.run(tf.nn.softmax(op_to_restore), feed_dict={x: x_test_image,'Placeholder:0':1.0})
            prob_maps.append(y_output[0,:,:,:])
        np.savez('WHS/Results/Predictions/region/train_prob_maps_axial_{}'.format(i),prob_maps=prob_maps)                            
    print("================ DONE WITH TRAIN PREDICTIONS! ==================")  

print("================ DONE WITH AXIAL PREDICTIONS! ==================")