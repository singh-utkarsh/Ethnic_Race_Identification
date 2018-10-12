# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:39:50 2018

@author: Akshay
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:22:51 2018

@author: Akshay
"""
# test set generator having equal classes
import numpy as np
import os
import random
import pickle
import re
def validation_data(loc,n,num_class):
        l=os.listdir(loc)
        for i,j in enumerate(l):
            
            l[i]=[j,re.findall(r"\D+",j)[0]]
        coll=pickle.load(open(r"F:\datasets\cv_corpus_v1\sorted_no_of_data2.plk","rb")) 
    	 
        print(coll[0:num_class])	
        rand=[list(random.sample(range(coll[4][1]),n)) for i in range(num_class)]
        val_X=[]
        val_y=[]
        counter=[0]*num_class
        pop=[]
    #        print(rand)
        for i,j in enumerate(l):
            z=np.zeros(num_class)
            for k,g in enumerate(coll[:num_class]):
                if g[0]==j[1]:
                    z[k]=1    
                    break
            class_no=np.argmax(z)
            
    #            print(class_no)
            if len(rand[class_no])>0 and min(rand[class_no])==counter[class_no] :
                val_X.append(pickle.load(open(r"{0}\{1}".format(loc,j[0]),"rb"))[0])
                val_y.append(z)
                rand[class_no].remove(min(rand[class_no]))
                pop.append(j[0])
                
            counter[class_no]+=1    
        
        
        return val_X,val_y,pop
           
        
            
    
