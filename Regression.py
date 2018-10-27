import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

result=np.zeros((50,90));
cost =np.zeros((2,90));
avgCost =np.zeros((2,90));
for j in range(50):
    print("j=",j)
    n=0
    for i in range(500,45001,500):
        idx = np.random.randint(45000, size=i)
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
        lin_reg.fit(X_train,y_train)
        y_pred=lin_reg.predict(X_test)
    
        cost1=np.sum(np.power(y_pred-y_test,2))/(2*y_test.size)
        print (i," ",cost1)
       
        cost[1,n]=cost1;
        n=n+1;
    result[j]=cost[1,:]
print(result)
avgCost=np.mean(result, axis=0)
index = np.arange(len(avgCost))

fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(index, avgCost)
#ax.plot(cost[0,:], avgCost, 'r', label='Cost')
plt.xticks(index, cost[0,:], fontsize=5, rotation=30)
ax.set_xlabel('Number Of Data', fontsize=5)
ax.set_ylabel('Cost Function', fontsize=5)
ax.legend(loc=2)
fig.savefig('AverageCostOver50Iterations.png')