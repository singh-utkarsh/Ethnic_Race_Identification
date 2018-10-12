# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:28:00 2018

@author: Akshay
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:13:41 2018

@author: Akshay
"""
from tensorflow import keras
import tensorflow as tf
import val_generator
import numpy as np
import batch
import os
import checkpoint_drive

# data
x_val,y_val,val=val_generator.validation_data(r"resize mel spec",1512,5)
c=batch.batch_gen(224*2,r"resize mel spec",[96,128,3],[5],5,val)
def val_g(x,y,batch):
    batch_features=np.zeros([batch,*[96,128,3]])
    batch_target=np.zeros([batch,5])
    i=1
    while true:
        if batch*i >len(x):
            i=1
        alpha=x[batch*i-batch:batch*i]
        beta=y[batch*i-batch:batch*i]
        batch_features[:len(alpha)]=alpha
        batch_target[:len(beta)]=beta
        i=i+1
        yield batch_features,batch_target
            
        
        
        
#The model
model=keras.models.Sequential()
Conv2d=keras.layers.Conv2D
GRU=keras.layers.GRU
LSTM=keras.layers.LSTM
Dropout=keras.layers.Dropout
pooling=keras.layers.MaxPooling2D
l2=keras.regularizers.l2
BN=keras.layers.BatchNormalization
model.add(Conv2d(64,[7,1],padding="Same",input_shape=[96,128,3],activation="relu"))
model.add(BN())
model.add(pooling(pool_size=(2,1)))
model.add(Dropout(0.25))
model.add(Conv2d(64,[5,1],padding="Same",activation="relu"))
model.add(BN())
model.add(pooling(pool_size=(3,1)))
model.add(Dropout(0.25))
model.add(Conv2d(64,[4,1],padding="Same",activation="relu",kernel_regularizer=l2(0.001)))
model.add(BN())
model.add(pooling(pool_size=(4,1)))
model.add(Dropout(0.2))
model.add(Conv2d(96,[3,1],padding="Same",activation="relu",kernel_regularizer=l2(0.001)))
model.add(BN())
model.add(pooling(pool_size=(4,1)))
model.add(Dropout(0.2))
#model.add(Conv2d(96,[5,5],padding="Same",activation="relu"))
#model.add(BN())
#model.add(pooling(pool_size=(2,1)))
#model.summary()
model.add(keras.layers.Reshape([128,96]))
model.add(GRU(512,return_sequences=True,unroll=True,recurrent_dropout=0.1,kernel_regularizer=l2(0.001)))

model.add(GRU(512,return_sequences=True,unroll=True,recurrent_dropout=0.1,dropout=0.1,kernel_regularizer=l2(0.001)))

model.add(keras.layers.Reshape([128,512,1]))
model.add(pooling(pool_size=(128,1)))
model.add(keras.layers.Reshape([512]))


#model.add(keras.layers.Reshape([150,1]))
#model.add(GRU())

model.add(keras.layers.Dense(5,activation="softmax"))
model.summary()
model.compile("Adadelta",loss='categorical_crossentropy',metrics=["accuracy"])
checkpoint=keras.callbacks.ModelCheckpoint("checkpoints/{epoch}-{val_acc}.h5",monitor='val_acc',mode="auto",save_best_only=True,save_weights_only=True)

TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
training_model = model

tpu_model = tf.contrib.tpu.keras_to_tpu_model(training_model,strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
checkpoint_drives=checkpoint_drive.checkpoint_drive("1QpSSBYSDkMfnRKx3GOL_gFTvKwkgOWLS")
#earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
tpu_model.fit_generator(c,steps_per_epoch=64,epochs=400,callbacks=[checkpoint,checkpoint_drives],validation_data=(np.array(x_val).reshape(-1,96,128,3),np.array(y_val)))
tpu_model.save_weights("checkpoints/modelConv5gru100b20e300d5.h5")
