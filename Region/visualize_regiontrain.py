#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:43:07 2018

@author: Josefine
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join('.', '..')) 
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pylab as pylab
%matplotlib inline
#
a = np.load('Results/region_performance.npz')
dice2 = a['dice']
acc = a['acc']
spec = a['spec']
sens = a['sens']
rates = a['rates']

#params = {'legend.fontsize': 'x-large',
#          'figure.figsize': (15, 5),
#         'axes.labelsize': 'x-large',
#         'axes.titlesize':'x-large',
#         'xtick.labelsize':'x-large',
#         'ytick.labelsize':'x-large'}
#pylab.rcParams.update(params)

def epoch_average(a,size):
    new_a = []
    i = 0
    while i < len(a):
        val = np.mean(a[i:i+size])
        new_a.append(val)
        i+=size
    return new_a

dim = 128

# Load validation curves
n_test = int(2*dim)
acc_cor = epoch_average(np.load('Results/train_hist/region/valid_acc_cor.npy'),n_test)
loss_cor = epoch_average(np.load('Results/train_hist/region/valid_loss_cor.npy'),n_test)
acc_sag = epoch_average(np.load('Results/train_hist/region/valid_acc_sag.npy'),n_test)
loss_sag = epoch_average(np.load('Results/train_hist/region/valid_loss_sag.npy'),n_test)
acc_axial = epoch_average(np.load('Results/train_hist/region/valid_acc_axial.npy'),n_test)
loss_axial = epoch_average(np.load('Results/train_hist/region/valid_loss_axial.npy'),n_test)

acc_axial_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/valid_acc_axial.npy'),n_test)
loss_axial_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/valid_loss_axial.npy'),n_test)
acc_sag_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/valid_acc_sag.npy'),n_test)
loss_sag_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/valid_loss_sag.npy'),n_test)
acc_cor_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/valid_acc_cor.npy'),n_test)
loss_cor_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/valid_loss_cor.npy'),n_test)

acc_cor_noaug = epoch_average(np.load('Results/train_hist/region_noaug/valid_acc_cor.npy'),n_test)
loss_cor_noaug = epoch_average(np.load('Results/train_hist/region_noaug/valid_loss_cor.npy'),n_test)
acc_sag_noaug = epoch_average(np.load('Results/train_hist/region_noaug/valid_acc_sag.npy'),n_test)
loss_sag_noaug = epoch_average(np.load('Results/train_hist/region_noaug/valid_loss_sag.npy'),n_test)
acc_axial_noaug = epoch_average(np.load('Results/train_hist/region_noaug/valid_acc_axial.npy'),n_test)
loss_axial_noaug = epoch_average(np.load('Results/train_hist/region_noaug/valid_loss_axial.npy'),n_test)

acc_axial_nothing = epoch_average(np.load('Results/train_hist/region_nothing/valid_acc_axial.npy'),n_test)
loss_axial_nothing = epoch_average(np.load('Results/train_hist/region_nothing/valid_loss_axial.npy'),n_test)
acc_sag_nothing = epoch_average(np.load('Results/train_hist/region_nothing/valid_acc_sag.npy'),n_test)
loss_sag_nothing = epoch_average(np.load('Results/train_hist/region_nothing/valid_loss_sag.npy'),n_test)
acc_cor_nothing = epoch_average(np.load('Results/train_hist/region_nothing/valid_acc_cor.npy'),n_test)
loss_cor_nothing = epoch_average(np.load('Results/train_hist/region_nothing/valid_loss_cor.npy'),n_test)

plt.rcParams.update({'font.size': 13})

plt.figure(figsize=(8*4, 8*3))
plt.subplot(3,4,3)
matplotlib.ticker.MultipleLocator(0.01)
plt.plot(acc_cor, label = 'Both')
plt.plot(acc_cor_noaug, label = 'Only drop out')
plt.plot(acc_cor_nodrop, label = 'Only augmentation')
plt.plot(acc_cor_nothing, label = 'Nothing')
plt.legend(loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
axes = plt.gca()
axes.set_ylim([0.95,1])
plt.title('Validation accuracy coronal network')
plt.subplot(3,4,4)
plt.title('Validation loss coronal network')
plt.plot(loss_cor, label = 'Both')
plt.plot(loss_cor_noaug, label = 'Only drop out')
plt.plot(loss_cor_nodrop, label = 'Only augmentation')
plt.plot(loss_cor_nothing, label = 'Nothing')
plt.legend(loc='upper right')
axes = plt.gca()
axes.set_ylim([0.02,0.19])
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(3,4,7)
plt.plot(acc_sag, label = 'Both')
plt.plot(acc_sag_noaug, label = 'Only drop out')
plt.plot(acc_sag_nodrop, label = 'Only augmentation')
plt.plot(acc_sag_nothing, label = 'Nothing')
axes = plt.gca()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
axes.set_ylim([0.95,1])
plt.legend(loc='lower right')
plt.title('Validation accuracy sagittal network')
plt.subplot(3,4,8)
plt.title('Validation loss sigittal network')
plt.plot(loss_sag, label = 'Both')
plt.plot(loss_sag_noaug, label = 'Only drop out')
plt.plot(loss_sag_nodrop, label = 'Only augmentation')
plt.plot(loss_sag_nothing, label = 'Nothing')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
axes = plt.gca()
axes.set_ylim([0.02,0.19])

plt.subplot(3,4,11)
plt.plot(acc_axial, label = 'Both')
plt.plot(acc_axial_noaug, label = 'Only drop out')
plt.plot(acc_axial_nodrop, label = 'Only augmentation')
plt.plot(acc_axial_nothing, label = 'Nothing')
axes = plt.gca()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
axes.set_ylim([0.95,1])
plt.title('Validation accuracy axial network')
plt.legend(loc='lower right')
plt.subplot(3,4,12)
plt.plot(loss_axial, label = 'Both')
plt.plot(loss_axial_noaug, label = 'Only drop out')
plt.plot(loss_axial_nodrop, label = 'Only augmentation')
plt.plot(loss_axial_nothing, label = 'Nothing')
plt.title('Validation loss axial network')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
axes = plt.gca()
axes.set_ylim([0.02,0.19])


#% Load training curves
n_test = int(13*6*dim)
acc_cor = epoch_average(np.load('Results/train_hist/region/train_acc_cor.npy'),n_test)
loss_cor = epoch_average(np.load('Results/train_hist/region/train_loss_cor.npy'),n_test)
acc_sag = epoch_average(np.load('Results/train_hist/region/train_acc_sag.npy'),n_test)
loss_sag = epoch_average(np.load('Results/train_hist/region/train_loss_sag.npy'),n_test)
acc_axial = epoch_average(np.load('Results/train_hist/region/train_acc_axial.npy'),n_test)
loss_axial = epoch_average(np.load('Results/train_hist/region/train_loss_axial.npy'),n_test)

n_test = int(13*6*dim)
acc_axial_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/train_acc_axial.npy'),n_test)
loss_axial_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/train_loss_axial.npy'),n_test)
acc_sag_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/train_acc_sag.npy'),n_test)
loss_sag_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/train_loss_sag.npy'),n_test)
acc_cor_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/train_acc_cor.npy'),n_test)
loss_cor_nodrop = epoch_average(np.load('Results/train_hist/region_nodrop/train_loss_cor.npy'),n_test)

n_test = int(13*dim)
acc_cor_noaug = epoch_average(np.load('Results/train_hist/region_noaug/train_acc_cor.npy'),n_test)
loss_cor_noaug = epoch_average(np.load('Results/train_hist/region_noaug/train_loss_cor.npy'),n_test)
acc_sag_noaug = epoch_average(np.load('Results/train_hist/region_noaug/train_acc_sag.npy'),n_test)
loss_sag_noaug = epoch_average(np.load('Results/train_hist/region_noaug/train_loss_sag.npy'),n_test)
acc_axial_noaug = epoch_average(np.load('Results/train_hist/region_noaug/train_acc_axial.npy'),n_test)
loss_axial_noaug = epoch_average(np.load('Results/train_hist/region_noaug/train_loss_axial.npy'),n_test)

acc_axial_nothing = epoch_average(np.load('Results/train_hist/region_nothing/train_acc_axial.npy'),n_test)
loss_axial_nothing = epoch_average(np.load('Results/train_hist/region_nothing/train_loss_axial.npy'),n_test)
acc_sag_nothing = epoch_average(np.load('Results/train_hist/region_nothing/train_acc_sag.npy'),n_test)
loss_sag_nothing = epoch_average(np.load('Results/train_hist/region_nothing/train_loss_sag.npy'),n_test)
acc_cor_nothing = epoch_average(np.load('Results/train_hist/region_nothing/train_acc_cor.npy'),n_test)
loss_cor_nothing = epoch_average(np.load('Results/train_hist/region_nothing/train_loss_cor.npy'),n_test)

plt.subplot(3,4,1)
plt.plot(acc_cor, label = 'Both')
plt.plot(acc_cor_noaug, label = 'Only drop out')
plt.plot(acc_cor_nodrop, label = 'Only augmentation')
plt.plot(acc_cor_nothing, label = 'Nothing')
axes = plt.gca()
axes.set_ylim([0.95,1])
plt.legend(loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train accuracy coronal network')
plt.subplot(3,4,2)
plt.title('Train loss coronal network')
plt.plot(loss_cor, label = 'Both')
plt.plot(loss_cor_noaug, label = 'Only drop out')
plt.plot(loss_cor_nodrop, label = 'Only augmentation')
plt.plot(loss_cor_nothing, label = 'Nothing')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
axes = plt.gca()
axes.set_ylim([0,0.05])

plt.subplot(3,4,5)
plt.plot(acc_sag, label = 'Both')
plt.plot(acc_sag_noaug, label = 'Only drop out')
plt.plot(acc_sag_nodrop, label = 'Only augmentation')
plt.plot(acc_sag_nothing, label = 'Nothing')
plt.legend(loc='lower right')
axes = plt.gca()
axes.set_ylim([0.95,1])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train accuracy sagittal network')
plt.subplot(3,4,6)
plt.title('Train loss sigittal network')
plt.plot(loss_sag, label = 'Both')
plt.plot(loss_sag_noaug, label = 'Only drop out')
plt.plot(loss_sag_nodrop, label = 'Only augmentation')
plt.plot(loss_sag_nothing, label = 'Nothing')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
axes = plt.gca()
axes.set_ylim([0,0.05])

plt.subplot(3,4,9)
plt.plot(acc_axial, label = 'Both')
plt.plot(acc_axial_noaug, label = 'Only drop out')
plt.plot(acc_axial_nodrop, label = 'Only augmentation')
plt.plot(acc_axial_nothing, label = 'Nothing')
plt.legend(loc='lower right')
axes = plt.gca()
axes.set_ylim([0.95,1])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train accuracy axial network')
plt.subplot(3,4,10)
plt.plot(loss_axial, label = 'Both')
plt.plot(loss_axial_noaug, label = 'Only drop out')
plt.plot(loss_axial_nodrop, label = 'Only augmentation')
plt.plot(loss_axial_nothing, label = 'Nothing')
plt.title('Train loss axial network')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
axes = plt.gca()
axes.set_ylim([0,0.05])

