#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 11:12:18 2023

@author: yumouwei
"""

import numpy as np
import cv2
import os
from tqdm import tqdm

base_dir = base_dir = os.getcwd()


def load_data_from_dir(set_df, resize=False, dim=(256, 256)):
    '''
    Load images and masks into two N-D arrays
    resize: Boolean
    dim = (width, height)
    '''
    data_dir = os.path.join(base_dir, 'data')
    image_stack = []
    mask_stack = []
    for img, msk in tqdm(zip(set_df['image'], set_df['mask'])):
    #for index, row in tqdm(set_df.iterrows()): # iterrows() is significantly slower 
        # load image
        image = cv2.imread(os.path.join(data_dir, img))# cv2 uses BGR order
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # BGR -> RGB
        # load mask
        mask = cv2.imread(os.path.join(data_dir, msk), cv2.IMREAD_GRAYSCALE) # load as grayscale image

        if resize:
            image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)

        image_stack.append(image) 
        mask_stack.append(mask)

    image_stack = np.array(image_stack) / 255.  # What I should have done was to set dtype=uint8 and scale to 255 later
    mask_stack = np.expand_dims(np.array(mask_stack), -1)

    return image_stack, mask_stack

color_dict = {
    0: (128, 128, 128),  # Sky
    1: (128, 0, 0),      # Building
    2: (192, 192, 128),  # Pole
    3: (128, 64, 128),   # Road
    4: (0, 0, 192),      # Sidewalk
    5: (128, 128, 0),    # Tree
    6: (192, 128, 128),  # SignSymbol
    7: (64, 64, 128),    # Fence
    8: (64, 0, 128),     # Car
    9: (64, 64, 0),      # Pedestrian
    10: (0, 128, 192),   # Bicyclist
    255: (0, 0, 0)      # Void
}

def convert_indexed_to_rgb_masks(masks):
    '''
    Convert a stack of indexed masks to RGB masks using the colormap of the original paper.
    '''
    masks_rgb = np.zeros((*masks.shape[0:3], 3), dtype=np.uint8)

    for i in range(11):
        color = color_dict[i]
        masks_rgb[(masks == i)[:,:,:,0]] = color
 
    return masks_rgb

def convert_indexed_to_rgb_mask(mask):
    '''
    Convert a single indexed masks to RGB masks using the colormap of the original paper.
    '''
    mask_rgb = np.zeros((*mask.shape[0:2], 3), dtype=np.uint8)

    for i in range(11):
        color = color_dict[i]
        mask_rgb[(mask == i)[:,:,0]] = color
 
    return mask_rgb

##########
# Evaluation metrics

# Pixel accuracy (global average) -- agrees with method using confusion matrix
def evaluate_pixel_accuracy_global(true_masks, pred_masks):
    true = true_masks[true_masks != 255]
    pred = pred_masks[pred_masks != 255]
    return np.sum(true == pred) / np.prod(true.shape)

# Class-wise pixel accuracy
def evaluate_pixel_accuracy_by_class(true_masks, pred_masks, i_class):
    true_i = (true_masks[true_masks != 255] == i_class)
    pred_i = (pred_masks[pred_masks != 255] == i_class)
    return np.sum(pred_i[true_i]) / np.sum(true_i)

#IOU by class
def evaluate_jaccard_score_by_class(true_masks, pred_masks, i_class=0, smooth=1):
    '''
    Calculate jaccard score (iou) for class i
    true_masks, pred_masks: 3D tensor with shape = (images, width, height)
    '''
    y_true = (true_masks == i_class).astype('int')
    # if y_true.sum() == 0:
    #     return np.nan
    y_pred = (pred_masks == i_class).astype('int')
    intersection = np.sum(np.abs(y_true * y_pred))
    union = np.sum(y_true)+np.sum(y_pred)-intersection
    #return (intersection + smooth) / (union + smooth) # avoid 0/0 error
    return intersection / union