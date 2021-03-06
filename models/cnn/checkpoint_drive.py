# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:50:14 2018

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

class checkpoint_drive(keras.callbacks.Callback):
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