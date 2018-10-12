# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 23:01:19 2018

@author: Akshay
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:16:52 2018

@author: Akshay
"""

#mfcc creator

import pickle
import os
import re
import numpy as np
import cv2
import librosa.display as display
import librosa
import matplotlib.pyplot as plt
import pylab


def mfcc(num_mfcc,destination,v_folder=False,dataset=False,resize_audio=False,resize_mfcc=False,median=0,d=0):
    print(1)
    print(v_folder is True)
    if v_folder:
     print("alpha")
     if not resize_mfcc:
          l=os.listdir(r"{0}".format(v_folder))
#          f=os.listdir(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\test voice")
          l=d.copy()
          c=open(r"F:\datasets\cv_corpus_v1\sorted_no_of_data.plk","rb")
          coll=pickle.load(c)
          c.close()
          cou=0    
          for b,i in enumerate(l):
              for k in range(5):
                if coll[k][0]==re.findall(r"\D+",i[1])[0] :
                  cou+=1
                  print(cou)
                  z=np.zeros([median])
                  
                  sample,sr=librosa.load(r"{0}\{1}".format(i[0],i[1]))
                  if resize_audio:
                      if sample.shape[0]>median:
                          sample=sample[:median]
                      else:
                          z[:sample.shape[0]]=sample[:]
                          sample=z
                  for j in range(len(num_mfcc)):
#                      mfcc = librosa.feature.mfcc(y=sample,sr=sr, 
#                                             n_fft=int(0.025*sr),hop_length=int(0.01*sr),
#                                             n_mfcc=num_mfcc[j])
                                logs=librosa.amplitude_to_db(librosa.stft(sample, n_fft=int(0.025*sr),hop_length=int(0.01*sr)),ref=np.max)
                                pylab.axis('off') 
                                pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) 
                                display.specshow(logs, sr=sr, x_axis='time', y_axis='log')
                                
                                pylab.savefig(r"{0}\{2}\{1}".format(destination,coll[k][0]+str(cou),j+1), bbox_inches=None, pad_inches=0)
                                pylab.close() 
                                w=open(r"{0}\{2}\{1}".format(destination,cou,j+1),"wb")
                                pickle.dump([logs,i],w)
                                w.close()
                     
           
     if resize_mfcc:
          print("beta") 
          l=os.listdir(r"{0}".format(v_folder))
          
          l=list(set(l))
          c=open(r"C:\Users\Akshay\Desktop\New folder (2)\sorted_no_of_data.plk","rb")
          coll=pickle.load(c)
          c.close()
          cou=0    
          for b,i in enumerate(l):
              for k in range(5):
                if coll[k][0]==re.findall(r"\D+",i)[0] :
                  cou+=1
                  print(cou)
                  z=np.zeros([median])
                  
                  sample,sr=librosa.load(r"{0}\{1}".format(v_folder,i))
                  
                  for j in range(len(num_mfcc)):
                      mfcc = librosa.feature.mfcc(y=sample,sr=sr, 
                                             n_fft=int(0.025*sr),hop_length=int(0.01*sr),
                                             n_mfcc=num_mfcc[j])
                      resized=cv2.resize(mfcc,(median,num_mfcc[j]))
                      w=open(r"{0}\{2}\{1}".format(destination,cou,j+1),"wb")
                      pickle.dump([resized,i],w)
                      w.close()
                                       
                          
                          
           
              
    
    if dataset:
        data=open(r"{0}".format(dataset),"rb")
        dataset=pickle.load(data)
        d=[]
        for i in range(len(num_mfcc)):
            d.append([])
        counter=0
        for j in dataset:
            print(counter)
            counter+=1
            y,sr=j[0]
            for i in range(len(num_mfcc)):
              mfcc = librosa.feature.mfcc(y=y,sr=sr, 
                                     n_fft=int(0.025*sr),hop_length=int(0.01*sr),
                                     n_mfcc=num_mfcc[i])
              d[i].append([mfcc,j[1]])
#        w=open(r"{0}\{1}".format(destination,mfcc_name),"wb")
#        pickle.dump(data,w)
#        w.close()

d=pickle.load(open(r"F:\datasets\cv_corpus_v1\folder_name.plk","rb"))
mfcc(v_folder=r"F:\datasets\cv_corpus_v1\use",num_mfcc=[13],destination=r"F:\datasets\cv_corpus_v1\mel spectrogram not cliped",median=int(79910.0),resize_mfcc=False,d=d) 
coll=pickle.load(open(r"F:\datasets\cv_corpus_v1\sorted_no_of_data2.plk","rb"))  