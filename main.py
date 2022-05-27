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
import unet2 as u

def main():
    #datasetの読み込み
    y = np.load('PAS300data.npy')
    z = np.load('CGH300data.npy')

    #訓練データと評価用データで分ける
    X_train , X_test, Y_train, Y_test = y[:800],y[800:], z[:800], z[800:]
    
    for i in range(X_train.shape[0]):
        X_train[i,:,:,0]=(X_train[i,:,:,0]-np.min(X_train[i,:,:,0]))/(np.max(X_train[i,:,:,0])-np.min(X_train[i,:,:,0]))
        X_train[i,:,:,1]=(X_train[i,:,:,1]-np.min(X_train[i,:,:,1]))/(np.max(X_train[i,:,:,1])-np.min(X_train[i,:,:,1]))
        Y_train[i,:,:,0]=(Y_train[i,:,:,0]-np.min(Y_train[i,:,:,0]))/(np.max(Y_train[i,:,:,0])-np.min(Y_train[i,:,:,0]))
        Y_train[i,:,:,1]=(Y_train[i,:,:,1]-np.min(Y_train[i,:,:,1]))/(np.max(Y_train[i,:,:,1])-np.min(Y_train[i,:,:,1]))
    for i in range(X_test.shape[0]):
        X_test[i,:,:,0] = (X_test[i,:,:,0]-np.min(X_test[i,:,:,0]))/(np.max(X_test[i,:,:,0])-np.min(X_test[i,:,:,0]))
        X_test[i,:,:,1] = (X_test[i,:,:,1]-np.min(X_test[i,:,:,1]))/(np.max(X_test[i,:,:,1])-np.min(X_test[i,:,:,1]))
        Y_test[i,:,:,0] = (Y_test[i,:,:,0]-np.min(Y_test[i,:,:,0]))/(np.max(Y_test[i,:,:,0])-np.min(Y_test[i,:,:,0]))
        Y_test[i,:,:,1] = (Y_test[i,:,:,1]-np.min(Y_test[i,:,:,1]))/(np.max(Y_test[i,:,:,1])-np.min(Y_test[i,:,:,1]))
    
    #モデル学習
    model = u.UNet()

    print(model.summary())
    adam = tf.keras.optimizers.Adam(learning_rate=0.005)

    model.compile(optimizer='adam',
              loss= 'mean_squared_error',
              metrics=['mean_squared_error'])
    
    fit_callbacs = [
        callbacks.EarlyStopping(monitor='val_loss',
                                patience=5,
                                mode='min')
    ]

    history = model.fit(X_train,Y_train,epochs=20,batch_size = 10,validation_data=(X_test, Y_test),callbacks=fit_callbacs,)

    metrics = ['loss']
    fig=plt.figure(figsize=(10, 5))
    for i in range(len(metrics)):
        metric = metrics[i]
        plt.title(metric)
        plt_train = history.history[metric]
        plt_test = history.history['val_' + metric]
        plt.plot(plt_train, label='training_loss')
        plt.plot(plt_test, label='validation_loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid(which='major',alpha=0.6)
        plt.grid(which='minor',alpha=0.3)
        plt.legend()
        plt.show()
    
    model.save('Unet_300epoch100')


if __name__ == '__main__':
    main()