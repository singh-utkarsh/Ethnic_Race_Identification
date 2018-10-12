# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:25:49 2018

@author: Akshay

"""

from tensorflow import keras
import tensorflow as tf
import pickle
from sklearn import cross_validation
import matplotlib.pyplot as plt
#import val_generator
import numpy as np
import batch
import os
#import checkpoint_train
tf.keras.backend.clear_session()
# data
x_val,y_val,val=val_generator.validation_data(r"resize mel spec",1512,5)
c=batch.batch_gen(79,r"F:\datasets\cv_corpus_v1\test mel spec pickled resized audio",[96,128,3],[5],5,[])
#The model
def val_g(x,y,batch):
    batch_features=np.zeros([batch,*[96,128,3]])
    batch_target=np.zeros([batch,5])
    i=1
    while True:
        if batch*i >len(x):
            i=1
        alpha=x[batch*i-batch:batch*i]
        beta=y[batch*i-batch:batch*i]
        batch_features[:len(alpha)]=alpha
        batch_target[:len(beta)]=beta
        i=i+1
        yield batch_features,batch_target
            
        
model=keras.models.Sequential()
Conv2d=keras.layers.Conv2D
Dropout=keras.layers.Dropout
pooling=keras.layers.MaxPooling2D
l2=keras.regularizers.l2
BN=keras.layers.BatchNormalization
model.add(Conv2d(32,[5,5],padding="Same",input_shape=[96,128,3],activation="relu",kernel_regularizer=l2(0.1)))
model.add(BN())
model.add(pooling(pool_size=(6,6)))
model.add(Conv2d(64,[5,5],padding="Same",activation="relu",kernel_regularizer=l2(0.01)))
model.add(BN())
model.add(pooling(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2d(64,[5,5],padding="Same",activation="relu"))
model.add(BN())
model.add(pooling(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2d(96,[5,5],padding="Same",activation="relu"))
model.add(BN())
model.add(pooling(pool_size=(2,1)))
model.add(Dropout(0.25))
#model.add(Conv2d(96,[5,5],padding="Same",activation="relu"))
#model.add(BN())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation="relu"))
model.add(Dropout(0.4))

#model.add(pooling(pool_size=(2,1)))
#model.add(keras.layers.Reshape([128,96,1]))
#model.add(pooling(pool_size=(1,96)))
#model.add(keras.layers.Reshape([128]))
#model.add(keras.layers.Reshape([150,1]))
#model.add(GRU())

model.add(keras.layers.Dense(5,activation="softmax"))
model.summary()
model.compile("Adadelta",loss='categorical_crossentropy',metrics=["accuracy"])

TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
training_model = model

tpu_model = tf.contrib.tpu.keras_to_tpu_model(training_model,strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))


checkpoint=keras.callbacks.ModelCheckpoint("checkpoints/{epoch}-{val_acc}.h5",monitor='val_acc',mode="auto",save_best_only=True)
#earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
#train_check=checkpoint_train.train_checkpoint("1D6-w_uRwxMn2aJRmT_7YxW4aNefEJ7vG")
tpu_model.fit_generator(c,steps_per_epoch=73,epochs=400,callbacks=[checkpoint],validation_data=(np.array(x_val).reshape(-1,96,128,3),np.array(y_val)))
tpu_model.save_weights("checkpoints/modelConv5.h5")

#prediction
model.load_weights(r"F:\common voice deep leaning\cnn audio cliped\conv4 layer\255-0.8878307311623185.h5")
evaluate=model.evaluate_generator(c,steps=5*7,max_queue_size=10,workers=0)