import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
n=0
cost =np.zeros((2,40));
for i in range(500,20001,500):
    idx = np.random.randint(20000, size=i)
    data=np.genfromtxt('CASP.csv', delimiter = ',')
    X=data[idx,1:9]
    y=data[idx,0]
    cost[0,n]=i;
   
    
#X=np.genfromtxt('DataSetLetter.csv', delimiter = ',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
   # print(X)
#y=np.genfromtxt('DataSetLetter.csv', delimiter = ',', dtype='str',usecols=(0))
   # print(y)
    X_train, X_test, y_train,y_test  = train_test_split(X,y,test_size=.2, random_state=0)
    lin_reg=LinearRegression()
    lin_reg.fit(X,y)
    y_pred=lin_reg.predict(X_test)

    cost1=np.sum(np.power(y_pred-y_test,2))/(2*y.size)
    print (i," ",cost1)
   
    cost[1,n]=cost1;
    n=n+1;

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(cost[0,:], cost[1,:], 'r', label='Cost')
ax.set_xlabel('Number Of Data')
ax.set_ylabel('Cost Function')
ax.legend(loc=2)
fig.savefig('pred.png')