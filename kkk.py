import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import feature_selection


    #for i in range(500,45001,500):
        #idx = np.random.randint(45000, size=i)
idx = np.random.randint(45000, size=45000)
data=np.genfromtxt('CASP.csv', delimiter = ',')
#X=data[idx,1:9]
#y=data[idx,0]
   # cost[0,n]=i;
X=data[idx,1:9]
y=data[idx,0]# -*- coding: utf-8 -*-



model = feature_selection.SelectKBest(score_func=feature_selection.mututal_info_regression ,k=4)
