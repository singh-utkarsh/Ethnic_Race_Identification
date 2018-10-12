# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 22:47:01 2018

@author: Akshay
"""

import keras
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
import os 
from google.colab import auth
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
class train_checkpoint(keras.callbacks.Callback):
    def __init__(self,ids):
        self.ids=ids
        self.used=[]
    def on_train_begin(self, logs={}):
        pass
 
    def on_train_end(self, logs={}):
        pass
 
    def on_epoch_begin(self,epoch, logs={}):
        pass       
 
    def on_epoch_end(self, epoch, logs={}):
        self.model.save("checkpoints/modelConv5gru100b20e300d5({0}).h5".format(str(epoch)+"-acc{0}".format(logs.get("acc"))))
        l=os.listdir("checkpoints")
         
        for i in list(set(l)-set(self.used)):
              f = drive.CreateFile({"parents": [{
              "kind": "drive#childList",
              "id": "{0}".format(self.ids)}] })
              f.SetContentFile("checkpoints/" +i )
              f.Upload()
        self.used=l.copy()   
     
    def on_batch_begin(self, batch, logs={}):
        pass
 
    def on_batch_end(self, batch, logs={}):
        pass