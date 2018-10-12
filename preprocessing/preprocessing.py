# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:02:19 2018

@author: Akshay
"""

import pandas as pd
import re
import os,shutil
import pickle
import librosa
import numpy as np
import random
import cv2
import pylab
import librosa.display as display
import matplotlib.pyplot as plt
plt.ioff()
data=pd.read_csv("cv-valid-train.csv")
k=data.dropna(subset=["accent"])
a=k["filename"]
b=k["accent"]
a=a.values
b=b.values
d={}
for i in range(len(a)):
    if b[i] not in d:
        d[b[i]]=1
    else:
        d[b[i]]+=1
        
    shutil.copy(a[i],r"use 2\{0}{1}.mp3".format(b[i],d[b[i]]))    
text=k["text"]
accent=k["accent"]
text=text.values
accent=accent.values
import collections
d=collections.defaultdict(str)
for i in range(len(text)):
    d[accent[i]]+=text[i]
    
for i in  d.keys():
    d[i]=collections.Counter(d[i].split())
    
    
a=open(r"F:\datasets\cv_corpus_v1\sorted_no_of_data.plk",'rb')

coll=pickle.load(a)

#median of common voice
length=[]
l2=[[r"F:\datasets\cv_corpus_v1\use",x] for x in os.listdir(r"F:\datasets\cv_corpus_v1\use")]+[[r"F:\datasets\cv_corpus_v1\use 2",x] for x in os.listdir(r"F:\datasets\cv_corpus_v1\use 2")]
c=open(r"F:\datasets\cv_corpus_v1\sorted_no_of_data2.plk","rb")
coll=pickle.load(c)

#editing the l2
l=[[],[],[],[],[]]
for i,k in enumerate(coll[0:5]):
    for j in l2:
        if re.findall(r"\D+",j[1])[0]==k[0]:
            l[i].append(j)
for i in range(5):
    l[i]=list(random.sample(l[i],7247))
d=[]
for i in l:
    d=d+i
count=0
for i in d:
    for j in coll[0:5]:
        if re.findall(r"\D+",i[1])[0]==j[0]:
            sample,sr=librosa.core.load(r"{0}\{1}".format(i[0],i[1]))
            mfcc = librosa.feature.mfcc(y=sample,sr=sr, 
                                         n_fft=int(0.025*sr),hop_length=int(0.01*sr),
                                         n_mfcc=13)
            save=open(r"F:\datasets\cv_corpus_v1\mfcc\{0}.plk".format(str(j[0])+str(count)),"wb")
            pickle.dump([mfcc,i],save)
            save.close()
            length.append(sample.shape)
            count+=1
            print(count,j[0],i)
            break
median=np.median(length) 
pickle.dump(median,open("medianofaudio.plk","wb"))
pickle.dump(d,open("folder_name.plk","wb"))

d=collections.Counter(k["accent"])
pickle.dump(d,open("d1.plk","wb"))
import subprocess
import re
arr=[]
for j,i in enumerate(os.listdir(r"F:\datasets\cv_corpus_v1\use 2")):
    process = subprocess.Popen(['ffmpeg',  '-i', r"F:\datasets\cv_corpus_v1\use 2\{0}".format(i)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    matches = re.search(r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),", str(stdout), re.DOTALL).groupdict()
    arr.append([i,matches["seconds"]])
    print(j)
round(len(sample)/sr,2)
#arr
arr1=[[x[0],float(x[1])] for x in arr]
import collections
coll=[re.findall(r"\D+",x)[0] for x in l2]
z=collections.Counter(coll)
z2=[x for x in zip(z.keys(),z.values())]
z2=sorted(z2,key=lambda x: x[1],reverse=True)
pickle.dump(z2,open("sorted_no_of_data2.plk","wb"))
med=[]
alpha=os.listdir(r"F:\datasets\cv_corpus_v1\mel spectrogram not cliped\1")
for i in alpha:
    data=pickle.load(open(r"F:\datasets\cv_corpus_v1\mel spectrogram not cliped\1\{0}".format(i),"rb"))
    med.append(data[0].shape)
    print(i)
med1=np.median(med,axis=0) 
alpha=os.listdir(r"F:\datasets\cv_corpus_v1\mel spectrogram not cliped\1")
for i in alpha:
        j=open(r"F:\datasets\cv_corpus_v1\mel spectrogram not cliped\1\{0}".format(i),"rb")
        data=pickle.load(j)
        j.close()
        data[0]=cv2.resize(data[0],(364,276))
        k=open("F:\datasets\cv_corpus_v1\mel spec not cliped resize\{0}".format(i),"wb")
        pickle.dump(data,k)
        k.close()
alpha=os.listdir(r"F:\datasets\cv_corpus_v1\mel spectrogram not cliped\1")
a=os.listdir(r"F:\datasets\cv_corpus_v1\alpha")
a=[x[:-3]+"mp3" for x in a]

for j,i in enumerate(set(alpha)):
    
    k=open(r"F:\datasets\cv_corpus_v1\mel spectrogram not cliped\1\{}".format(i),"rb")
    data=pickle.load(k)
    k.close()
    
    
    logs=data[0]
    pylab.axis('off') 
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) 
    display.specshow(logs, sr=22050, x_axis='time', y_axis='log')
    
    pylab.savefig(r"{0}\{1}.jpg".format(r"F:\datasets\cv_corpus_v1\alpha",data[1][1][:-4]+"("+str(j)+")"), bbox_inches=None, pad_inches=0)
    pylab.close() 
    print(j)

names=[x for x in names2 if a[x]==2]
import collections
a=collections.Counter(names2)