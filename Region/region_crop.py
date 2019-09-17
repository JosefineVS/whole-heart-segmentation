#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:14:27 2018

@author: Josefine
"""

import numpy as np
import re
import nibabel as nib
import glob
from skimage.transform import resize

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

# Create original high res data function:
def create_data(filename_img):
    a = nib.load(filename_img)
    a = a.get_data()
    a2 = np.clip(a,-1000,1000)
    a3 = np.interp(a2, (a2.min(), a2.max()), (-1, +1))
    # Reshape:
    img = np.zeros([512,512,512])+np.min(a3)
    index1 = int(np.ceil((512-a.shape[2])/2))
    index2 = int(512-np.floor((512-a.shape[2])/2))
    img[:,:,index1:index2] = a3
    images = img.transpose((2,0,1))
    return images

def create_label(filename_label):
    # Label creation
    b = nib.load(filename_label)
    b = b.get_data()
    img = np.zeros([b.shape[0],b.shape[0],b.shape[0]])
    index1 = int(np.ceil((img.shape[2]-b.shape[2])/2))
    index2 = int(img.shape[2]-np.floor((img.shape[2]-b.shape[2])/2))
    img[:,:,index1:index2] = b
    labels = img.transpose((2,0,1))
    return labels

# Fusion of low resolution probablity maps
def fusion(prob_maps_axial, prob_maps_sag, prob_maps_cor):
    # Reshape sagittal data to match axial:
    sag_to_axial = prob_maps_sag.transpose((2, 0, 1, 3))
    # Reshape coronal data to match axial:
    cor_to_sag = prob_maps_cor.transpose((1, 0, 2, 3))
    cor_to_axial = cor_to_sag.transpose((2, 0, 1, 3))
    temp = np.maximum.reduce([sag_to_axial,cor_to_axial,prob_maps_axial])
    return temp

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
filelist_test = natural_sort(glob.glob('WHS/ct_train_test/ct_test/*_image.nii.gz')) # list of file names

filelist_train = natural_sort(glob.glob('WHS/Augment_data/*_image.nii')) # list of file names
filelist_train_label = natural_sort(glob.glob('WHS/Augment_data/*_label.nii')) # list of file names

# Load test data:
files_p0_axial = natural_sort(glob.glob('WHS/Results/Predictions/region/test_prob_maps_axial_*.npz')) # list of file names
files_p0_sag = natural_sort(glob.glob('WHS/Results/Predictions/region/test_prob_maps_sag_*.npz')) # list of file names
files_p0_cor = natural_sort(glob.glob('WHS/Results/Predictions/region/test_prob_maps_cor_*.npz')) # list of file names

## Load train data:
files_p1_axial = natural_sort(glob.glob('WHS/Results/Predictions/region/train_prob_maps_axial_*.npz')) # list of file names
files_p1_sag = natural_sort(glob.glob('WHS/Results/Predictions/region/train_prob_maps_sag_*.npz')) # list of file names
files_p1_cor = natural_sort(glob.glob('WHS/Results/Predictions/region/train_prob_maps_cor_*.npz')) # list of file names

#for n in range(len(files_p0_axial)):
#    axial_data = np.load(files_p0_axial[n])
#    prob_maps_axial = axial_data['prob_maps']
#    sag_data = np.load(files_p0_sag[n])
#    prob_maps_sag = sag_data['prob_maps']
#    cor_data = np.load(files_p0_cor[n])
#    prob_maps_cor = cor_data['prob_maps']
#
#    # Create fused propability map
#    fused_prob_maps = fusion(prob_maps_axial, prob_maps_sag, prob_maps_cor)
#    full_prob_maps = np.zeros([512,512,512,2])
#    for i in range(2):
#        full_prob_maps[:,:,:,i] = resize(fused_prob_maps[:,:,:,i],(512,512,512))    
#    label = full_prob_maps.argmax(axis=-1)
#    image = create_data(filelist_test[n])
#
#    # Get bounding box
#    i,i2,j,j2,k,k2 = cut_region(label)
#    # Load original data
#    factor =int(np.ceil(0.02*image.shape[0]))
#    start = int(np.floor(np.min([i,j,k])-factor))
#    end = int(np.ceil(np.max([i2,j2,k2])+factor))
#    cut = [start,end]
#    if cut[0] < 0:
#        cut[0] = 0
#    if cut[1] > image.shape[0]:
#        cut[1] = image.shape[0]
#    # Crop bounding box of original data
#    cut_img = image[cut[0]:cut[1],cut[0]:cut[1],cut[0]:cut[1]]
#    np.savez('WHS/Data/test_segments_{}'.format(n),images=cut_img,cut=cut)
#    print('Test image', (n+1), 'cut', (cut))

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
    image = create_data(filelist_train[n])
    groundtruth = create_label(filelist_train_label[n])
    # Get bounding box
    i,i2,j,j2,k,k2 = cut_region(labels)

    # Load original data
    factor =int(np.ceil(0.02*groundtruth.shape[0]))
    mult_factor = image.shape[0]/labels.shape[0]
    start = int(np.floor(np.min([i,j,k])*mult_factor-factor))
    end = int(np.ceil(np.max([i2,j2,k2])*mult_factor+factor))
    cut = [start,end]
    if cut[0] < 0:
        cut[0] = 0
    if cut[1] > image.shape[0]:
        cut[1] = image.shape[0]
    # Crop bounding box of original data
    cut_GT = groundtruth[cut[0]:cut[1],cut[0]:cut[1],cut[0]:cut[1]]
    cut_GT = np.round(cut_GT)
    cut_img = image[cut[0]:cut[1],cut[0]:cut[1],cut[0]:cut[1]]
    np.savez('WHS/Data/train_segments_{}'.format(n),images=cut_img,labels=cut_GT,cut=cut)
    print('Train image', (n+1), 'cut', (cut))
