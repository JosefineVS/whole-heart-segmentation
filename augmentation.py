#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:46:04 2019

@author: Josefine
"""

import re
import glob
import sitk_functions as func
import SimpleITK as sitk

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

filename_img = natural_sort(glob.glob('WHS/ct_train_test/ct_train/*_image.nii.gz')) # list of file names
filename_label = natural_sort(glob.glob('WHS/ct_train_test/ct_train/*_label.nii.gz')) # list of file names

for k in range(len(filename_img)):
    print('Load image',(k+1))
    img_sitk = sitk.ReadImage(filename_img[k])
    label_sitk = sitk.ReadImage(filename_label[k])
    filter = sitk.MinimumMaximumImageFilter()
    filter.Execute(img_sitk)
    min_val = filter.GetMinimum()
    
    # NORMAL
    sitk.WriteImage(img_sitk,'WHS/Augment_data/normal_{}_image.nii'.format(k))
    sitk.WriteImage(label_sitk,'WHS/Augment_data/normal_{}_label.nii'.format(k))

#    # ROTATION
#    [img_rot, label_rot] = func.affine_rotate(img_sitk,label_sitk,min_val)
#    sitk.WriteImage(img_rot,'WHS/Augment_data/rot_{}_image.nii'.format(k))
#    sitk.WriteImage(label_rot,'WHS/Augment_data/rot_{}_label.nii'.format(k))

#    # SHEAR
#    [img_sh, label_sh] = func.affine_shear(img_sitk,label_sitk,min_val)
#    sitk.WriteImage(img_sh,'WHS/Augment_data/sh_{}_image.nii'.format(k))
#    sitk.WriteImage(label_sh,'WHS/Augment_data/sh_{}_label.nii'.format(k))
#    
#    # Intensity
#    l1 = func.mult_and_add_intensity_fields(img_sitk)
#    sitk.WriteImage(l1,'WHS/Augment_data/intensity_{}_image.nii'.format(k))
#    sitk.WriteImage(label_sitk,'WHS/Augment_data/intensity_{}_label.nii'.format(k))

#    # B SPLINE
    numcontrolpoints = 5
    stdDeform = 15
    dim = 3
    [img_bspline, label_bspline] = func.BSplineDeform(img_sitk,label_sitk, dim, numcontrolpoints, stdDeform,min_val)
    sitk.WriteImage(img_bspline,'WHS/Augment_data/bspline_{}_image.nii'.format(k))
    sitk.WriteImage(label_bspline,'WHS/Augment_data/bspline_{}_label.nii'.format(k))

    # B SPLINE
    numcontrolpoints = 10
    stdDeform = 10
    dim = 3
    [img_bspline, label_bspline] = func.BSplineDeform(img_sitk,label_sitk, dim, numcontrolpoints, stdDeform,min_val)
    sitk.WriteImage(img_bspline,'WHS/Augment_data/bspline2_{}_image.nii'.format(k))
    sitk.WriteImage(label_bspline,'WHS/Augment_data/bspline2_{}_label.nii'.format(k))
    print('Finished with image',(k+1))
