#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:22:42 2018

@author: Josefine
"""

import numpy as np
import re
import tensorflow as tf
import nibabel as nib
import glob
from skimage.transform import resize

imgDim = 128
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

def create_groundtruth(filename_label,direction):
    labels = []
    b = nib.load(filename_label)
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
    labels_onehot = np.squeeze(np.stack((labels==0, labels==1), axis=3).astype('int32'))
    return labels_onehot

# Create original high res data function:
def create_data(filename_img,filename_label):
    images = []
    a = nib.load(filename_img)
    a = a.get_data()
    a2 = np.clip(a,-1000,1000)
    a3 = np.interp(a2, (a2.min(), a2.max()), (-1, +1))
    # Reshape:
    img = np.zeros([512,512,512])+np.min(a3)
    index1 = int(np.ceil((512-a.shape[2])/2))
    index2 = int(512-np.floor((512-a.shape[2])/2))
    img[:,:,index1:index2] = a3
    for i in range(img.shape[2]):
            images.append((img[:,:,i]))
    images = np.asarray(images)

    # Label creation
    labels = []
    b = nib.load(filename_label)
    b = b.get_data()
    img = np.zeros([b.shape[0],b.shape[0],b.shape[0]])
    index1 = int(np.ceil((img.shape[2]-b.shape[2])/2))
    index2 = int(img.shape[2]-np.floor((img.shape[2]-b.shape[2])/2))
    img[:,:,index1:index2] = b
    for i in range(img.shape[2]):
            labels.append((img[:,:,i]))            
    labels = np.asarray(labels)

    return images, labels

# Fusion of low resolution probablity maps
def fusion(prob_maps_axial, prob_maps_sag, prob_maps_cor):
    # Reshape sagittal data to match axial:
    sag_to_axial = []
    for i in range(prob_maps_sag.shape[2]):
        sag_to_axial.append((prob_maps_sag[:,:,i,:]))  
    sag_to_axial = np.asarray(sag_to_axial)
    # Reshape coronal data to match axial:
    cor_to_sag = []
    for i in range(prob_maps_cor.shape[2]):
        cor_to_sag.append((prob_maps_cor[:,i,:,:]))  
    cor_to_sag = np.asarray(cor_to_sag)
    cor_to_axial = []
    for i in range(prob_maps_cor.shape[2]):
        cor_to_axial.append((cor_to_sag[:,:,i,:]))  
    cor_to_axial = np.asarray(cor_to_axial)
    temp = np.maximum.reduce([sag_to_axial,cor_to_axial,prob_maps_axial])
    return temp

# Compute accuracy and dice:
def performance(original_label_axial,labels):
    correct_prediction = np.equal(original_label_axial, labels)
    accuracy = np.count_nonzero(correct_prediction)/correct_prediction.size

    im1 = np.asarray(original_label_axial).astype(np.bool)
    im2 = np.asarray(labels).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    dice = 2. * intersection.sum() / (im1.sum() + im2.sum())
    
    TP = (np.sum(np.logical_and(labels == 1, original_label_axial == 1))/(512*512*512))*100
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = (np.sum(np.logical_and(labels == 0, original_label_axial == 0))/(512*512*512))*100
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = (np.sum(np.logical_and(labels == 1, original_label_axial == 0))/(512*512*512))*100
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = (np.sum(np.logical_and(labels == 0, original_label_axial == 1))/(512*512*512))*100
    rates = [TP, TN, FP, FN]
    sens = TP/(FN+TP)
    spec = TN/(FP+TN)
    jacc = TP / (TP + FP + FN)
    return accuracy, dice, rates, sens, spec, jacc

# Region retraction
def cut_region(volumen1):
    for i in range(volumen1.shape[0]):
        if np.max(volumen1[i,:,:]) == 1:
            break    
    
    for j in range(volumen1.shape[1]):
        if np.max(volumen1[:,j,:]) == 1:
            break    
        
    for k in range(volumen1.shape[2]):
        if np.max(volumen1[:,:,k]) == 1:
            break
        
    for i2 in reversed(range(volumen1.shape[0])):
        if np.max(volumen1[i2,:,:]) == 1:
            break    
    
    for j2 in reversed(range(volumen1.shape[1])):
        if np.max(volumen1[:,j2,:]) == 1:
            break    
        
    for k2 in reversed(range(volumen1.shape[2])):
        if np.max(volumen1[:,:,k2]) == 1:
            break    
    #factor = int(np.ceil(0.02*volumen1.shape[0]))
    #cut_volumen = volumen1[i-factor:i2+factor,j-factor:j2+factor,k-factor:k2+factor]
    return i,i2,j,j2,k,k2

# Load data:
filelist_val = natural_sort(glob.glob('WHS/validation/*_image.nii.gz')) # list of file names
filelist_val_label = natural_sort(glob.glob('WHS/validation/*_label.nii.gz')) # list of file names

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

    for i in range(len(filelist_val)):
        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_val)))+1))
        # Find renderings corresponding to the given name
        prob_maps = []
        x_test = create_image(filelist_val[i],'axial')
        y_test = create_groundtruth(filelist_val_label[i],'axial')
        for k in range(x_test.shape[0]):
            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
            y_output = sess.run(tf.nn.softmax(op_to_restore), feed_dict={x: x_test_image,'Placeholder:0':1.0})
            prob_maps.append(y_output[0,:,:,:])
        np.savez('WHS/Results/Predictions/region/val_prob_maps_axial_{}'.format(i),prob_maps=prob_maps)                            
print("================ DONE WITH AXIAL PREDICTIONS! ==================")

##############################################################################
###                  Reload network and predict                         ######
##############################################################################
print("====================== LOAD CORONAL NETWORK: ===========================")

# Doing predictions with the model 
tf.reset_default_graph()      

new_saver = tf.train.import_meta_graph('WHS/Results/region/model_cor/model.ckpt.meta')

with tf.Session() as sess:
    new_saver.restore(sess, tf.train.latest_checkpoint('WHS/Results/region/model_cor/'))
    graph = tf.get_default_graph()       
    x = graph.get_tensor_by_name("x_train:0")
    op_to_restore = graph.get_tensor_by_name("output/Softmax:0") #ME

    for i in range(len(filelist_val)):
        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_val)))+1))
        # Find renderings corresponding to the given name
        prob_maps = []
        x_test = create_image(filelist_val[i],'cor')
        y_test = create_groundtruth(filelist_val_label[i],'cor')
        for k in range(x_test.shape[0]):
            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
            y_output = sess.run(tf.nn.softmax(op_to_restore), feed_dict={x: x_test_image,'Placeholder:0':1.0})
            prob_maps.append(y_output[0,:,:,:])
        np.savez('WHS/Results/Predictions/region/val_prob_maps_cor_{}'.format(i),prob_maps=prob_maps)                            
print("================ DONE WITH CORONAL PREDICTIONS! ==================")

##############################################################################
###                  Reload network and predict                         ######
##############################################################################
print("====================== LOAD SAGITTAL NETWORK: ===========================")

# Doing predictions with the model 
tf.reset_default_graph()      

new_saver = tf.train.import_meta_graph('WHS/Results/region/model_sag/model.ckpt.meta')

with tf.Session() as sess:
    new_saver.restore(sess, tf.train.latest_checkpoint('WHS/Results/region/model_sag/'))
    graph = tf.get_default_graph()       
    x = graph.get_tensor_by_name("x_train:0")
    op_to_restore = graph.get_tensor_by_name("output/Softmax:0") #ME

    for i in range(len(filelist_val)):
        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_val)))+1))
        # Find renderings corresponding to the given name
        prob_maps = []
        x_test = create_image(filelist_val[i],'sag')
        y_test = create_groundtruth(filelist_val_label[i],'sag')
        for k in range(x_test.shape[0]):
            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)
            y_output = sess.run(tf.nn.softmax(op_to_restore), feed_dict={x: x_test_image,'Placeholder:0':1.0})
            prob_maps.append(y_output[0,:,:,:])
        np.savez('WHS/Results/Predictions/region/val_prob_maps_sag_{}'.format(i),prob_maps=prob_maps)                            
print("================ DONE WITH SAGITTAL PREDICTIONS! ==================")

# Load test data:
files_p1_axial = natural_sort(glob.glob('WHS/Results/Predictions/region/val_prob_maps_axial_*.npz')) # list of file names
files_p1_sag = natural_sort(glob.glob('WHS/Results/Predictions/region/val_prob_maps_sag_*.npz')) # list of file names
files_p1_cor = natural_sort(glob.glob('WHS/Results/Predictions/region/val_prob_maps_cor_*.npz')) # list of file names

for n in range(len(files_p1_axial)):
    axial_data = np.load(files_p1_axial[n])
    prob_maps_axial = axial_data['prob_maps']
    sag_data = np.load(files_p1_sag[n])
    prob_maps_sag = sag_data['prob_maps']
    cor_data = np.load(files_p1_cor[n])
    prob_maps_cor = cor_data['prob_maps']

    # Create fused propability map
    fused_prob_maps = fusion(prob_maps_axial, prob_maps_sag, prob_maps_cor)
    labels = fused_prob_maps.argmax(axis=-1)
    image, groundtruth = create_data(filelist_val[n],filelist_val_label[n])

    # Get bounding box
    i,i2,j,j2,k,k2 = cut_region(labels)

    # Load original data
    factor =int(np.ceil(0.02*groundtruth.shape[0]))
    mult_factor = image.shape[0]/labels.shape[0]
    start = int(np.floor(np.min([i,j,k])*mult_factor-factor))
    end = int(np.ceil(np.max([i2,j2,k2])*mult_factor+factor))
    cut = [start,end]
    # Crop bounding box of original data
    cut_GT = groundtruth[start:end,start:end,start:end]
    cut_img = image[start:end,start:end,start:end]
    np.savez('WHS/Data/validation_segments_{}'.format(n),images=cut_img,labels=cut_GT,cut=cut)
    print('Train image', (n+1))
