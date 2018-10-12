# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:16:52 2018

@author: Akshay
"""

#mfcc creator
import librosa
import pickle
import os
import re
import numpy as np
import cv2
import librosa.display as display
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
                  
                      if sample.shape[0]>median:
                          sample=sample[:median]
                      else:
                          z[:sample.shape[0]]=sample[:]
                          sample=z
                  for j in range(len(num_mfcc)):
                      mfcc = librosa.feature.mfcc(y=sample,sr=sr, 
                                             n_fft=int(0.025*sr),hop_length=int(0.01*sr),
                                             n_mfcc=num_mfcc[j])
                      w=open(r"{0}\{2}\{1}".format(destination,coll[k][0]+str(cou),j+1),"wb")
                      pickle.dump([mfcc,i],w)
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
mfcc(v_folder=r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram",mfcc_name="non_aug_cliped_padded3264128.plk",num_mfcc=[32,64,128],destination=r"C:\Users\Akshay\Desktop\New folder (2)\datasets")        
#data=open(),"rb")
#dataset=data    
"""" resizing the mfcc"""" #make a function regarding this so as to use in the future """      
k=open(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\non_aug_3264128_mfcc_resized.plk","rb")
data=pickle.load(k)
k.close()  
med=[[],[],[]]
for i,g in enumerate(data):
    for j in g:
        med[i].append(j[0].shape[1])
med1=np.median(med,axis=1) 
for i,g in enumerate(data):
    for j,d in enumerate(g):
        resized=cv2.resize(d[0],(2358,d[0].shape[0]))
        data[i][j][0]=resized        

k=open(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\non_aug_cliped_padded3264128_mfcc.plk","wb")
pickle.dump(data,k)
k.close()

k=open(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\non_aug_cliped_padded.plk","rb")
d=pickle.load(k)

l=os.listdir(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\test voice")

med=[]
for i in d :
    if i[1] not in l:
        med.append(i[0][0].shape)
med1=np.median(med) 
mfcc(v_folder=r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram",mfcc_name="non_aug_cliped_padded32+64+128_audio_cliped_padded.plk",num_mfcc=[32,64,128],destination=r"C:\Users\Akshay\Desktop\New folder (2)\datasets",median=int(med1)) 

k=open(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\non_aug_cliped_padded32+64+128_audio_cliped_padded.plk","rb")
data=pickle.load(k)
k.close()
for i,d  in enumerate(data):
    for j,k in enumerate(d):
        with open(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\non_aug_cliped_padded32+64+128_audio_cliped_padded\{1}\{0}.plk".format(j,i+1),"wb") as f:
            pickle.dump(k,f)
da=[]            
for i in range(100):
    d=open(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\non_aug_cliped_padded32+64+128_audio_cliped_padded\{1}\{0}.plk".format(i,2),"rb")  
    f=pickle.load(d)
    da.append(f)
mfcc(v_folder=r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram",num_mfcc=[13],destination=r"C:\Users\Akshay\Desktop\New folder (2)\datasets\true_train_Data_aug_cliped_pad",median=int(med1)) 
##median of aug data
length=[]
l2=os.listdir(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram")
l=os.listdir(r"C:\Users\Akshay\Desktop\New folder (2)\datasets\test voice")
c=open(r"C:\Users\Akshay\Desktop\New folder (2)\sorted_no_of_data.plk","rb")
aug=os.listdir(r"C:\Users\Akshay\Desktop\New folder (2)\augmented")
coll=pickle.load(c)
count=0
for i in list(set(l2)-set(l)-set(aug)):
    for j in coll[0:5]:
        if re.findall(r"\D+",i)[0]==j[0]:
            sample,sr=librosa.load(r"{0}\{1}".format(r"C:\Users\Akshay\Desktop\New folder (2)\spectrogram",i))
#            mfcc = librosa.feature.mfcc(y=sample,sr=sr, 
#                                         n_fft=int(0.025*sr),hop_length=int(0.01*sr),
#                                         n_mfcc=13)
            length.append(sample.shape)
            count+=1
            print(count)
            break
median=np.median(length)    
mfcc(v_folder=r"F:\datasets\cv_corpus_v1\use",num_mfcc=[13],destination=r"F:\datasets\cv_corpus_v1\mfcc cliped and pad",median=int(79910.0),resize_mfcc=False,d=d) 
#median of common voice
length=[]
l2=os.listdir(r"F:\datasets\cv_corpus_v1\mel spectrogram\1")
c=open(r"F:\datasets\cv_corpus_v1\sorted_no_of_data2.plk","rb")
coll=pickle.load(c)
count=0
for i in list(set(l2)):
    for j in coll[0:5]:
        if re.findall(r"\D+",i)[0]==j[0]:
            d=cv2.imread(r"F:\datasets\cv_corpus_v1\mel spectrogram not cliped\1\{}".format(i))
            resized=cv2.resize(d,(int(d.shape[1]/5),int(d.shape[0]/5)))
            save=open(r"F:\datasets\cv_corpus_v1\resize mel spec not cliped\{0}".format(str(j[0])+str(count)),"wb")
            pickle.dump([resized,j[0]],save)
            save.close()
            
            count+=1
            print(count)
            break
median=np.median(length) 
for i,g in enumerate(data):
    for j,d in enumerate(g):
        resized=cv2.resize(d[0],(2358,d[0].shape[0]))
        data[i][j][0]=resized 