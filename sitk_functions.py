#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 12:23:12 2018

@author: Josefine
"""
import numpy as np
import random
import SimpleITK as sitk

def affine_rotate(img_sitk,label_sitk,min_value):
    rotate = random.choice([-15,-10,10,15])
    new_transform = sitk.AffineTransform(3)    
    new_transform = sitk.AffineTransform(new_transform)
    matrix = np.array(new_transform.GetMatrix()).reshape((3,3))
    radians_z = -np.pi * rotate / 180.
    rotation = np.array([[np.cos(radians_z), -np.sin(radians_z),0],[np.sin(radians_z), np.cos(radians_z),0],[0,0,1]])
    new_matrix = np.dot(rotation, matrix)
    new_transform.SetMatrix(new_matrix.ravel())
    interpolator = sitk.sitkNearestNeighbor
    img_rot = sitk.Resample(img_sitk, new_transform,interpolator,min_value)
    label_rot = sitk.Resample(label_sitk, new_transform,interpolator,0)
    return [img_rot,label_rot]

def affine_shear(img_sitk,label_sitk,min_value):
    shear = (random.uniform(-0.1,0.1), random.uniform(-0.1,0.1),random.uniform(-0.1,0.1))
    new_transform = sitk.AffineTransform(3)
    new_transform = sitk.AffineTransform(new_transform)
    matrix = np.array(new_transform.GetMatrix()).reshape((3,3))
    matrix[0,1] = -shear[0]
    matrix[1,0] = -shear[1]
    matrix[0,2] = -shear[1]
    matrix[2,0] = -shear[0]
    matrix[1,2] = -shear[2]
    matrix[2,1] = -shear[2]
    new_transform.SetMatrix(matrix.ravel())
    interpolator = sitk.sitkNearestNeighbor
    img_sh = sitk.Resample(img_sitk, new_transform,interpolator,min_value)
    label_sh = sitk.Resample(label_sitk, new_transform,interpolator,0)
    return [img_sh,label_sh]

def mult_and_add_intensity_fields(original_image):
    '''
    Modify the intensities using multiplicative and additive Gaussian bias fields.
    '''
    # Gaussian image with same meta-information as original (size, spacing, direction cosine)
    # Sigma is half the image's physical size and mean is the center of the image. 
    sigma = random.uniform(0.5,3)
    middel = random.uniform(1,3)
    g_mult = sitk.GaussianSource(original_image.GetPixelIDValue(),
                             original_image.GetSize(),
                             [(sz-1)*spc/sigma for sz, spc in zip(original_image.GetSize(), original_image.GetSpacing())],
                             original_image.TransformContinuousIndexToPhysicalPoint(np.array(original_image.GetSize())/middel),
                             255,
                             original_image.GetOrigin(),
                             original_image.GetSpacing(),
                             original_image.GetDirection())

    # Gaussian image with same meta-information as original (size, spacing, direction cosine)
    # Sigma is 1/8 the image's physical size and mean is at 1/16 of the size
    sigma = random.uniform(2,8)
    middel = random.uniform(4,10)
    g_add = sitk.GaussianSource(original_image.GetPixelIDValue(),
                             original_image.GetSize(),
               [(sz-1)*spc/sigma for sz, spc in zip(original_image.GetSize(), original_image.GetSpacing())],
               original_image.TransformContinuousIndexToPhysicalPoint(np.array(original_image.GetSize())/middel),
               255,
               original_image.GetOrigin(),
               original_image.GetSpacing(),
               original_image.GetDirection())
    l1 = g_mult*original_image+g_add
    return l1

def BSplineDeform(img, lab, dim, numcontrolpoints, stdDeform,min_value):
    transfromDomainMeshSize=[numcontrolpoints]*dim
    tx = sitk.BSplineTransformInitializer(img,transfromDomainMeshSize)
    params = tx.GetParameters()
    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDeform
    paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad
    params=tuple(paramsNp)
    tx.SetParameters(params)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) # ORIGINAL: sitk.sitkLinear
    resampler.SetDefaultPixelValue(min_value)
    resampler.SetTransform(tx)
    resampler.SetDefaultPixelValue(min_value) # -1024 HU on CT image = air
    outimgsitk = resampler.Execute(img)
    resampler.SetDefaultPixelValue(0)
    outlabsitk = resampler.Execute(lab)
    return outimgsitk, outlabsitk