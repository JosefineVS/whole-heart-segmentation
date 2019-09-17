#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 19:44:39 2018

@author: Josefine
"""

import numpy as np
import re
import nibabel as nib
import glob
from skimage.transform import resize
from scipy import ndimage

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

# Create original high res data function:
def create_data(filename_img,filename_label):
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
    for i in range(img.shape[2]):
            images.append((img[:,:,i]))
    images = np.asarray(images)

    return images

# Fusion of low resolution probablity maps
def fusion(prob_maps_axial, prob_maps_sag, prob_maps_cor):
    sag_to_axial = []
    for i in range(prob_maps_sag.shape[2]):
        sag_to_axial.append((prob_maps_sag[:,i,:,:]))  
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
    cor_to_axial2 = []
    for i in range(prob_maps_cor.shape[2]):
        cor_to_axial2.append((cor_to_axial[:,i,:,:]))  
    cor_to_axial = np.asarray(cor_to_axial2)
    
    temp = np.maximum.reduce([sag_to_axial,cor_to_axial,prob_maps_axial])
    return temp

def remove_objects(binary_mask):
    labelled_mask, num_labels = ndimage.label(binary_mask)

    # Let us now remove all the too small regions.
    refined_mask = binary_mask.copy()
    minimum_cc_sum = 5000
    for label in range(num_labels):
        if np.sum(refined_mask[labelled_mask == label]) < minimum_cc_sum:
            refined_mask[labelled_mask == label] = 0
    return refined_mask

filelist_train = natural_sort(glob.glob('WHS/ct_train_test/ct_test/*_image.nii.gz')) # list of file names
cropped_files = natural_sort(glob.glob('WHS/Data/test_segments_*.npz')) # list of file names

files_axial = natural_sort(glob.glob('WHS/Results/Predictions/segment/train_prob_maps_axial_*.npz')) # list of file names
files_sag = natural_sort(glob.glob('WHS/Results/Predictions/segment/train_prob_maps_sag_*.npz')) # list of file names
files_cor = natural_sort(glob.glob('WHS/Results/Predictions/segment/train_prob_maps_cor_*.npz')) # list of file names

for n in range(len(files_axial)):
    axial_data = np.load(files_axial[n])
    prob_maps_axial = axial_data['prob_maps']
    sag_data = np.load(files_sag[n])
    prob_maps_sag = sag_data['prob_maps']
    cor_data = np.load(files_cor[n])
    prob_maps_cor = cor_data['prob_maps']
    cut_file = np.load(cropped_files[n])
    cut = cut_file['cut']

    # Create fused propability map
    fused_prob_maps = fusion(prob_maps_axial, prob_maps_sag, prob_maps_cor)
    side_length = cut[1]-cut[0]
    lab = np.zeros([side_length,side_length,side_length,8])
    for i in range(8):
        lab[:,:,:,i] = resize(fused_prob_maps[:,:,:,i],(side_length,side_length,side_length))
    full_labels = np.zeros([512,512,512,8])
    full_labels[cut[0]:cut[1],cut[0]:cut[1],cut[0]:cut[1],:] = lab
    labels = full_labels.argmax(axis=-1)
    print('Test image', (n+1))
    np.savez('WHS/Results/Predictions/final/prediction_{}'.format(n),prob_map = labels)
