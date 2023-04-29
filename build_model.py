#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 11:12:39 2023

@author: yumouwei
"""

from tensorflow.keras import layers, Model, Input

activation = 'gelu'
#upconv_kernel_size = (3,3)
upconv_kernel_size = (2,2)

def Conv2DBlock(input_tensor, filters, kernel_size, USE_BN = True):
    '''
    Build a block of 2 Conv2D layers for U-Net encoder & decoder
    Return block output
    '''
    x = layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    if USE_BN:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if USE_BN:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def build_unet(input_shape, num_classes, num_filters = 16, kernel_size = 3):
    '''
    Build a U-Net model
    Return model
    '''
    x = inputs = Input(input_shape)

    # Build encoder
    e1 = Conv2DBlock(x, num_filters * 1, kernel_size)
    p1 = layers.MaxPooling2D(2, padding='same')(e1)

    e2 = Conv2DBlock(p1, num_filters * 2, kernel_size)
    p2 = layers.MaxPooling2D(2, padding='same')(e2)

    e3 = Conv2DBlock(p2, num_filters * 4, kernel_size)
    p3 = layers.MaxPooling2D(2, padding='same')(e3)

    e4 = Conv2DBlock(p3, num_filters * 8, kernel_size)
    p4 = layers.MaxPooling2D(2, padding='same')(e4)

    latent = Conv2DBlock(p4, num_filters * 16, kernel_size)

    # Build decoder
    d4 = layers.Conv2DTranspose(num_filters * 8, upconv_kernel_size, 2, padding='same', activation=activation)(latent)
    d4 = layers.concatenate([d4, e4])
    d4 = Conv2DBlock(d4, num_filters * 8, kernel_size)

    d3 = layers.Conv2DTranspose(num_filters * 4, upconv_kernel_size, 2, padding='same', activation=activation)(d4)
    d3 = layers.concatenate([d3, e3])
    d3 = Conv2DBlock(d3, num_filters * 4, kernel_size)

    d2 = layers.Conv2DTranspose(num_filters * 2, upconv_kernel_size, 2, padding='same', activation=activation)(d3)
    d2 = layers.concatenate([d2, e2])
    d2 = Conv2DBlock(d2, num_filters * 4, kernel_size)

    d1 = layers.Conv2DTranspose(num_filters * 1, upconv_kernel_size, 2, padding='same', activation=activation)(d2)
    d1 = layers.concatenate([d1, e1])
    d1 = Conv2DBlock(d1, num_filters * 1, kernel_size)

    outputs = layers.Conv2D(num_classes, 1, padding="same", activation = "softmax")(d1)
    model = Model(inputs, outputs, name='U-Net')
    return model

def build_unet_resnet50v2(encoder, num_classes, num_filters = 32, kernel_size = 3):
    '''
    Build a U-Net model with a pre-trained resnet50v2 as the encoder
    Return model
    
    U-Net blocks (filters=32): (224, 224, 32) -> (112, 112, 64) -> (56, 56, 128) -> (28, 28, 256) -> (14, 14, 512)
    resnet50v2 skip connection layers:
      - 'conv1_conv'           output shape: (112, 112, 64)
      - 'conv2_block2_out'    output shape: (56, 56, 256)
      - 'conv3_block3_out'     output shape: (28, 28, 512)
      - 'conv4_block5_out'     output shape: (14, 14, 1024)
    encoder output: 
      - 'post_relu'            output shape: (7, 7, 2048)
    '''
    encoder.trainable = False
    inputs = encoder.input
    latent = encoder.output     # latent layer, dim = (7, 7, 2048)
    
    # Build decoder
    # Decoder block 1 - (14, 14, 512)
    d1 = layers.Conv2DTranspose(num_filters * 16, upconv_kernel_size, 2, padding='same', activation=activation)(latent)
    skip1 = encoder.get_layer('conv4_block5_out').output
    d1 = layers.concatenate([d1, skip1])
    d1 = Conv2DBlock(d1, num_filters * 16, kernel_size)
    
    # Decoder block 2 - (28, 28, 256)
    d2 = layers.Conv2DTranspose(num_filters * 8, upconv_kernel_size, 2, padding='same', activation=activation)(d1)
    skip2 = encoder.get_layer('conv3_block3_out').output
    d2 = layers.concatenate([d2, skip2])
    d2 = Conv2DBlock(d2, num_filters * 8, kernel_size)    
    
    # Decoder block 3 - (56, 56, 128)
    d3 = layers.Conv2DTranspose(num_filters * 4, upconv_kernel_size, 2, padding='same', activation=activation)(d2)
    skip3 = encoder.get_layer('conv2_block2_out').output
    d3 = layers.concatenate([d3, skip3])
    d3 = Conv2DBlock(d3, num_filters * 4, kernel_size)
    
    # Decoder block 4 - (112, 112, 64)
    d4 = layers.Conv2DTranspose(num_filters * 2, upconv_kernel_size, 2, padding='same', activation=activation)(d3)
    skip4 = encoder.get_layer('conv1_conv').output
    d4 = layers.concatenate([d4, skip4])
    d4 = Conv2DBlock(d4, num_filters * 2, kernel_size)  

    # Decoder block 5 - (224, 224, 32)  -- no skip connection for this block because it'd be the input layer!
    d5 = layers.Conv2DTranspose(num_filters * 1, upconv_kernel_size, 2, padding='same', activation=activation)(d4)
    skip5 = encoder.get_layer('input_1').output
    d5 = layers.concatenate([d5, skip5])
    d5 = Conv2DBlock(d5, num_filters * 1, kernel_size)  
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation = "softmax")(d5)
    model = Model(inputs, outputs, name='U-Net-ResNet50V2')
    return model

def build_unet_mobilenetv2(encoder, num_classes, num_filters = 32, kernel_size = 3):
    '''
    Build a U-Net model with a pre-trained resnet50v2 as the encoder
    Return model
    
    reference: https://www.kaggle.com/code/mistag/train-keras-u-net-mobilenetv2/notebook
    U-Net blocks (filters=32): (224, 224, 32) -> (112, 112, 64) -> (56, 56, 128) -> (28, 28, 256) -> (14, 14, 512)
    MobileNetV2 skip connection layers:
      - 'block_1_expand_relu'        output shape: (112, 112, 96)
      - 'block_3_expand_relu'        output shape: (56, 56, 144)
      - 'block_6_expand_relu'        output shape: (28, 28, 192)
      - 'block_13_expand_relu'       output shape: (14, 14, 576)
    encoder output: 
      - 'out_relu'            output shape: (7, 7, 1280)
    '''
    encoder.trainable = False
    inputs = encoder.input
    latent = encoder.output     # latent layer, dim = (7, 7, 2048)
    
    # Build decoder
    # Decoder block 1 - (14, 14, 512)
    d1 = layers.Conv2DTranspose(num_filters * 16, upconv_kernel_size, 2, padding='same', activation=activation)(latent)
    skip1 = encoder.get_layer('block_13_expand_relu').output
    d1 = layers.concatenate([d1, skip1])
    d1 = Conv2DBlock(d1, num_filters * 16, kernel_size)
    
    # Decoder block 2 - (28, 28, 256)
    d2 = layers.Conv2DTranspose(num_filters * 8, upconv_kernel_size, 2, padding='same', activation=activation)(d1)
    skip2 = encoder.get_layer('block_6_expand_relu').output
    d2 = layers.concatenate([d2, skip2])
    d2 = Conv2DBlock(d2, num_filters * 8, kernel_size)    
    
    # Decoder block 3 - (56, 56, 128)
    d3 = layers.Conv2DTranspose(num_filters * 4, upconv_kernel_size, 2, padding='same', activation=activation)(d2)
    skip3 = encoder.get_layer('block_3_expand_relu').output
    d3 = layers.concatenate([d3, skip3])
    d3 = Conv2DBlock(d3, num_filters * 4, kernel_size)
    
    # Decoder block 4 - (112, 112, 64)
    d4 = layers.Conv2DTranspose(num_filters * 2, upconv_kernel_size, 2, padding='same', activation=activation)(d3)
    skip4 = encoder.get_layer('block_1_expand_relu').output
    d4 = layers.concatenate([d4, skip4])
    d4 = Conv2DBlock(d4, num_filters * 2, kernel_size)  

    # Decoder block 5 - (224, 224, 32)  -- no skip connection for this block because it'd be the input layer!
    d5 = layers.Conv2DTranspose(num_filters * 1, upconv_kernel_size, 2, padding='same', activation=activation)(d4)
    skip5 = encoder.get_layer('input_1').output
    d5 = layers.concatenate([d5, skip5])
    d5 = Conv2DBlock(d5, num_filters * 1, kernel_size)  
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation = "softmax")(d5)
    model = Model(inputs, outputs, name='U-Net-MobileNetV2')
    return model