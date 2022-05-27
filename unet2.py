import tensorflow as tf
from tensorflow import keras
from keras.layers import Input ,Activation,BatchNormalization,Conv2D,MaxPooling2D,UpSampling2D,Reshape,Dropout,Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import callbacks
import random
import math
import diffraction_algos as dalgo
import numpy as np
import matplotlib.pyplot as plt

#ニューラルネットワーク(U-Net)
def UNet():
    ##256x256x2
    inputs = Input(shape=(256,256,2))
    
    ##256x256x64
    z1 = Conv2D(64,(3,3),name='block1_conv1',activation='relu',padding = 'same')(inputs)
    z1 = Conv2D(64,(3,3),name='block1_conv2',padding = 'same') (z1)
    z1 = BatchNormalization()(z1)
    z1 = Activation('relu')(z1)
    z1_pool = MaxPooling2D((2,2), strides=None, name='block1_pool')(z1)
    
    ##128x128x128
    z2 = Conv2D(128,(3,3),name='block2_conv1',activation='relu',padding = 'same')(z1_pool)
    z2 = Conv2D(128,(3,3) , name='block2_conv2', padding = 'same')(z2)
    z2 = BatchNormalization()(z2)
    z2 = Activation('relu')(z2)
    z2_pool = MaxPooling2D((2,2), strides=None, name='block2_pool')(z2)
    
    ##64x64x256
    z3 = Conv2D(256, (3, 3) , name='block3_conv1', activation = 'relu', padding = 'same')(z2_pool)
    z3 = Conv2D(256, (3, 3) , name='block3_conv2', padding = 'same')(z3)
    z3 = BatchNormalization()(z3)
    z3 = Activation('relu')(z3)
    z3_pool = MaxPooling2D((2,2), strides=None, name='block3_pool')(z3)
    
    ##32x32x512
    z4 = Conv2D(512, (3, 3) , name='block4_conv1', activation = 'relu', padding = 'same')(z3_pool)
    z4 = Conv2D(512, (3, 3) , name='block4_conv2', padding = 'same')(z4)
    z4 = BatchNormalization()(z4)
    z4 = Activation('relu')(z4)
    z4_dropout = Dropout(0.5)(z4)
    z4_pool = MaxPooling2D((2,2), strides=None, name='block4_pool')(z4_dropout)
    
    #16x16x1024
    z5 = Conv2D(1024, (3, 3) , name='block5_conv1', activation = 'relu', padding = 'same')(z4_pool)
    z5 = Conv2D(1024, (3, 3) , name='block5_conv2', padding = 'same')(z5)
    z5 = BatchNormalization()(z5)
    z5 = Activation('relu')(z5)
    z5_dropout = Dropout(0.5)(z5)
    
    ##32x32x512
    z6_up = UpSampling2D((2,2))(z5_dropout)
    z6 = Conv2D(512, (2, 2) , name='block6_conv1', activation = 'relu', padding = 'same')(z6_up)
    z6 = Concatenate()([z4_dropout,z6])
    z6 = Conv2D(512, (3, 3) , name='block6_conv2', activation = 'relu', padding = 'same')(z6)
    z6 = Conv2D(512, (3, 3) , name='block6_conv3', padding = 'same')(z6)
    z6 = BatchNormalization()(z6)
    z6 = Activation('relu')(z6)
    
    ##64x64x256
    z7_up = UpSampling2D((2,2))(z6)
    z7 = Conv2D(256, (2, 2) , name='block7_conv1', activation = 'relu', padding = 'same')(z7_up)
    #z7 = concatenate([z3,z7], axis =3)
    z7 = Concatenate()([z3,z7])
    z7 = Conv2D(256, (3, 3) , name='block7_conv2', activation = 'relu', padding = 'same')(z7)
    z7 = Conv2D(256, (3, 3) , name='block7_conv3', padding = 'same')(z7)
    z7 = BatchNormalization()(z7)
    z7 = Activation('relu')(z7)
    
    ##128x128x128
    z8_up = UpSampling2D((2,2))(z7)
    z8 = Conv2D(128, (2, 2) , name='block8_conv1', activation = 'relu', padding = 'same')(z8_up)
    #z8 = concatenate([z2,z8], axis = 3)
    z8 = Concatenate()([z2,z8])
    z8 = Conv2D(128, (3, 3) , name='block8_conv2', activation = 'relu', padding = 'same')(z8)
    z8 = Conv2D(128, (3, 3) , name='block8_conv3', padding = 'same')(z8)
    z8 = BatchNormalization()(z8)
    z8 = Activation('relu')(z8)
    
    ##256x256x64
    z9_up = UpSampling2D((2,2))(z8)
    z9 = Conv2D(64, (2, 2) , name='block9_conv1', activation = 'relu', padding = 'same')(z9_up)
    #z9 = concatenate([z1,z9], axis = 3)
    z9 = Concatenate()([z1,z9])
    z9 = Conv2D(64, (3, 3) , name='block9_conv2', activation = 'relu', padding = 'same')(z9)
    z9 = Conv2D(64, (3, 3) , name='block9_conv3', padding = 'same')(z9)
    z9 = BatchNormalization()(z9)
    z9 = Activation('relu')(z9)
    
    #256x256x2
    output = Conv2D(2, (1, 1), name='output_conv', activation = 'sigmoid')(z9)
    
    model = Model(inputs = inputs, outputs = output)
    return model