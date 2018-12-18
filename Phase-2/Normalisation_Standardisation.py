# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 04:50:52 2018

@author: advancerajat
"""

import numpy as np

newX=np.zeros((30000,23))
X=np.genfromtxt('credit_card.csv', delimiter = ',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23))

#min1=np.min(X, axis=0)
##max1=np.max(X, axis=0)
#for i in range(max1.size):
#    if((max1[i] - min1[i]) != 0) :
#        newX[:,i]=(X[:,i] - min1[i])/(max1[i] - min1[i])
#    else :
#        newX[:,i]=X[:,i]
               
mean1=np.mean(X, axis=0)
print(mean1)
std1=np.std(X, axis=0)
print(std1)
for i in range(23):
    if((std1[i]) != 0) :
        newX[:,i]=(X[:,i] - mean1[i])/(std1[i])
    else :
        newX[:,i]=X[:,i]
        
print(newX.mean(axis=0))
np.savetxt("updated_creditcard_dataset_standadrise.csv", newX, delimiter=",")