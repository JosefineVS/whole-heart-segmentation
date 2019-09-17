# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 08:57:40 2019

@author: josh
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
files_pred = natural_sort(glob.glob('Results/nii/*_label.nii.gz'))

for n in range(len(files_pred)):
    b0 = nib.load(files_pred[n])
    new_header_b = b0.header.copy()   # Lav en kopi af den header den skal være magen til (i dette tilfælde lab)
    b = b0.get_data()
    volume = np.load(files_pred[n])
    data = volume['images']   
    label = volume['labels']   

    labBox = nib.Nifti1Image(data, b0.affine, new_header_b)
    labBox3 = nib.Nifti1Image(label, b0.affine, new_header_b)

    index = n + 1
    nib.save(labBox3,'WHS/Results/nii/test{}_label.nii.gz'.format(index))
    nib.save(labBox,'WHS/Results/nii/test{}.nii.gz'.format(index))