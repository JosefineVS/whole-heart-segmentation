#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:36:59 2019

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

def remove_objects(binary_mask):
    labelled_mask, num_labels = ndimage.label(binary_mask)

    # Let us now remove all the too small regions.
    refined_mask = binary_mask.copy()
    minimum_cc_sum = 5000
    for label in range(num_labels):
        if np.sum(refined_mask[labelled_mask == label]) < minimum_cc_sum:
            refined_mask[labelled_mask == label] = 0
    return refined_mask

def performance(sep_truth,sep_classes):
    accuracy,dice_coeff = [],[]
    for i in range(sep_classes.shape[3]):
        correct_prediction = np.equal(sep_truth[:,:,:,i], sep_classes[:,:,:,i])
        acc = np.count_nonzero(correct_prediction)/correct_prediction.size
        accuracy.append(acc)
        y_true_f = np.ndarray.flatten(sep_truth[:,:,:,i])
        y_pred_f = np.ndarray.flatten(sep_classes[:,:,:,i])
        intersection = np.sum(y_true_f * y_pred_f)
        dice = (2 * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))
        dice_coeff.append(dice)
    return accuracy, dice_coeff

def performance_rate(gt,labels):
    TP = (np.sum([np.logical_and(labels == 1, gt == 500),np.logical_and(labels == 2, gt == 600),np.logical_and(labels == 3, gt == 420),np.logical_and(labels == 4, gt == 550),np.logical_and(labels == 5, gt == 205),np.logical_and(labels == 6, gt == 820),np.logical_and(labels == 7, gt == 850)])/(512*512*512))*100
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = (np.sum(np.logical_and(labels == 0, gt == 0))/(512*512*512))*100
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = (np.sum([np.logical_not(labels == 1, gt == 500),np.logical_not(labels == 2, gt == 600),np.logical_not(labels == 3, gt == 420),np.logical_not(labels == 4, gt == 550),np.logical_not(labels == 5, gt == 205),np.logical_not(labels == 6, gt == 820),np.logical_not(labels == 7, gt == 850)])/(512*512*512))*100
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = (np.sum(np.logical_not(labels == 0, gt == 0))/(512*512*512))*100
    rates = [TP, TN, FP, FN]
    sens = TP/(FN+TP)
    spec = TN/(FP+TN)
    return rates, sens, spec

filelist_train = natural_sort(glob.glob('Data/validation/*_image.nii')) # list of file names
filelist_train_label = natural_sort(glob.glob('Data/validation/*_label.nii')) # list of file names
cropped_files = natural_sort(glob.glob('Data/region/validation_segments_*.npz')) # list of file names

files_axial = natural_sort(glob.glob('Results/Predictions/segment/val_prob_maps_cor_*.npz')) # list of file names

accuracy, dice_coeff,rates_test,sens_test,spec_test = [],[],[],[],[]
for n in range(len(files_axial)):
    axial_data = np.load(files_axial[n])
    prob_maps_axial = axial_data['prob_maps']
    cut_file = np.load(cropped_files[n])
    cut = cut_file['cut']
    image, groundtruth = create_data(filelist_train[n],filelist_train_label[n])

    if cut[0] < 0:
        cut[0] = 0
    if cut[1] > image.shape[0]:
        cut[1] = image.shape[0]
    # Create fused propability map
    side_length = cut[1]-cut[0]
    lab = np.zeros([side_length,side_length,side_length,8])
    for i in range(8):
        lab[:,:,:,i] = resize(prob_maps_axial[:,:,:,i],(side_length,side_length,side_length))
    full_labels = np.zeros([512,512,512,8])
    full_labels[cut[0]:cut[1],cut[0]:cut[1],cut[0]:cut[1],:] = lab
    l = full_labels.argmax(axis=-1)
    labels = l.transpose((1, 0, 2))
    labels = labels.transpose((2, 0, 1)) 
    labels = labels.transpose((1, 0, 2)) 
    sep_classes = np.stack((labels==0, labels==1, labels==2, labels==3, labels ==4, labels==5, labels ==6, labels==7), axis=3)
    sep_truth = np.squeeze(np.stack((groundtruth==0, groundtruth==500, groundtruth==600, groundtruth==420, groundtruth ==550, groundtruth==205, groundtruth ==820, groundtruth==850), axis=3))
    sep_truth[:,:,:,0] = np.logical_not(sep_truth[:,:,:,0]).astype(int)
    sep_classes[:,:,:,0] = np.logical_not(sep_classes[:,:,:,0]).astype(int)
    
    acc, dice = performance(sep_truth,sep_classes)
    
    print('Test image', (n+1),'Dice',(dice))
    dice_coeff.append(dice)
    accuracy.append(acc)
    
np.savez('Results/train/segmentation_cor_val_performance', acc = accuracy,dice = dice_coeff)

files_axial = natural_sort(glob.glob('Results/Predictions/segment/val_prob_maps_sag_*.npz')) # list of file names

accuracy, dice_coeff,rates_test,sens_test,spec_test = [],[],[],[],[]
for n in range(len(files_axial)):
    axial_data = np.load(files_axial[n])
    prob_maps_axial = axial_data['prob_maps']
    cut_file = np.load(cropped_files[n])
    cut = cut_file['cut']
    image, groundtruth = create_data(filelist_train[n],filelist_train_label[n])

    if cut[0] < 0:
        cut[0] = 0
    if cut[1] > image.shape[0]:
        cut[1] = image.shape[0]

    # Create fused propability map
    side_length = cut[1]-cut[0]
    lab = np.zeros([side_length,side_length,side_length,8])
    for i in range(8):
        lab[:,:,:,i] = resize(prob_maps_axial[:,:,:,i],(side_length,side_length,side_length))
    full_labels = np.zeros([512,512,512,8])
    full_labels[cut[0]:cut[1],cut[0]:cut[1],cut[0]:cut[1],:] = lab
    l = full_labels.argmax(axis=-1)
    labels = l.transpose((1, 0, 2))
   
    sep_classes = np.stack((labels==0, labels==1, labels==2, labels==3, labels ==4, labels==5, labels ==6, labels==7), axis=3)
    sep_truth = np.squeeze(np.stack((groundtruth==0, groundtruth==500, groundtruth==600, groundtruth==420, groundtruth ==550, groundtruth==205, groundtruth ==820, groundtruth==850), axis=3))
    sep_truth[:,:,:,0] = np.logical_not(sep_truth[:,:,:,0]).astype(int)
    sep_classes[:,:,:,0] = np.logical_not(sep_classes[:,:,:,0]).astype(int)
    
    acc, dice = performance(sep_truth,sep_classes)
    
    print('Test image', (n+1),'Dice',(dice))
    dice_coeff.append(dice)
    accuracy.append(acc)

np.savez('Results/train/segmentation_sag_val_performance', acc = accuracy,dice = dice_coeff)

files_axial = natural_sort(glob.glob('Results/Predictions/segment/val_prob_maps_axial_*.npz')) # list of file names

accuracy, dice_coeff,rates_test,sens_test,spec_test = [],[],[],[],[]
for n in range(len(files_axial)):
    axial_data = np.load(files_axial[n])
    prob_maps_axial = axial_data['prob_maps']
    cut_file = np.load(cropped_files[n])
    cut = cut_file['cut']
    
    image, groundtruth = create_data(filelist_train[n],filelist_train_label[n])

    if cut[0] < 0:
        cut[0] = 0
    if cut[1] > image.shape[0]:
        cut[1] = image.shape[0]
    # Create fused propability map
    side_length = cut[1]-cut[0]
    lab = np.zeros([side_length,side_length,side_length,8])
    for i in range(8):
        lab[:,:,:,i] = resize(prob_maps_axial[:,:,:,i],(side_length,side_length,side_length))
    full_labels = np.zeros([512,512,512,8])
    full_labels[cut[0]:cut[1],cut[0]:cut[1],cut[0]:cut[1],:] = lab
    labels = full_labels.argmax(axis=-1)
    sep_classes = np.stack((labels==0, labels==1, labels==2, labels==3, labels ==4, labels==5, labels ==6, labels==7), axis=3)
    sep_truth = np.squeeze(np.stack((groundtruth==0, groundtruth==500, groundtruth==600, groundtruth==420, groundtruth ==550, groundtruth==205, groundtruth ==820, groundtruth==850), axis=3))
    sep_truth[:,:,:,0] = np.logical_not(sep_truth[:,:,:,0]).astype(int)
    sep_classes[:,:,:,0] = np.logical_not(sep_classes[:,:,:,0]).astype(int)
    
    acc, dice = performance(sep_truth,sep_classes)
    
    print('Test image', (n+1),'Dice',(dice))
    dice_coeff.append(dice)
    accuracy.append(acc)
    
    #np.savez('Results/Predictions/final_axial/prediction_{}'.format(n),prob_map = full_labels,dice=dice,acc=acc)
np.savez('Results/train/segmentation_axial_val_performance', acc = accuracy,dice = dice_coeff)
