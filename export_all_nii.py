#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:57:23 2019

@author: Josefine
"""

import numpy as np
import re
import glob
from  scipy import ndimage
import nibabel as nib

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

files_im = natural_sort(glob.glob('WHS/ct_train_test/ct_test/*_image.nii.gz')) # list of file names
files_pred = natural_sort(glob.glob('WHS/Results/Predictions/final/prediction_*.npz')) # list of file names

for n in range(len(files_pred)):
    b0 = nib.load(files_im[n])
    new_header_b = b0.header.copy()   # Lav en kopi af den header den skal være magen til (i dette tilfælde lab)

    volume = np.load(files_pred[n])
    data = volume['prob_map']   
    b = b0.get_data()
    index1 = int(np.ceil((512-b.shape[2])/2))
    index2 = int(512-np.floor((512-b.shape[2])/2))
    cut_data = data[index1:index2,:,:]
    pred = cut_data.transpose((1,2,0))

    # New CCA (Nearest neighbour interpolation)
    sep_classes = np.stack((pred==0, pred==1, pred==2, pred==3, pred==4, pred==5, pred==6, pred==7), axis=3)
    CCA_classes = np.zeros([pred.shape[0],pred.shape[1],pred.shape[2],8])
    error = np.zeros([pred.shape[0],pred.shape[1],pred.shape[2],8])
    for m in range(CCA_classes.shape[-1]): 
        labelled_mask, num_labels = ndimage.label(sep_classes[:,:,:,m])
        largest_cc_mask = (labelled_mask == (np.bincount(labelled_mask.flat)[1:].argmax() + 1))
        CCA_classes[:,:,:,m] = largest_cc_mask
        labelled_mask[largest_cc_mask] = 0
        labelled_mask[labelled_mask>0] = 1
        error[:,:,:,m] = labelled_mask
    classes = CCA_classes.argmax(axis=3)
    all_errors = error.argmax(axis=3)
    all_errors[all_errors > 0] = 1
    classes = classes.astype(np.float64)
    all_errors = all_errors.astype(np.float64)
    classes[all_errors == 1] = np.nan
    invalid = np.isnan(classes)
    ind = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    postproc = classes[tuple(ind)].astype('uint32')
    postproc[postproc==1] = 500
    postproc[postproc==2] = 600
    postproc[postproc==3] = 420
    postproc[postproc==4] = 550
    postproc[postproc==5] = 205
    postproc[postproc==6] = 820
    postproc[postproc==7] = 850
    labBox = nib.Nifti1Image(postproc, b0.affine, new_header_b)
    index = n + 1
    nib.save(labBox,'WHS/Results/nii/ct_test_20{}_label.nii.gz'.format(index))