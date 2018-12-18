import datetime
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
cost =np.zeros((3,90));
avgCost =np.zeros((2,90));
n=0
data=np.genfromtxt('CASP.csv', delimiter = ',')



       
        
    #X=np.genfromtxt('DataSetLetter.csv', delimiter = ',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
       # print(X)
    #y=np.genfromtxt('DataSetLetter.csv', delimiter = ',', dtype='str',usecols=(0))
       # print(y\\
for i in range(500,45000,500):
    cost[0,n]=i;
    idx = np.random.randint(45000, size=i)
       
    X=data[idx,1:]
    y=data[idx,0]
    X_train, X_test, y_train,y_test  = train_test_split(X,y,test_size=.2, random_state=0)
    starttime=datetime.datetime.now()
    print("starttime",starttime)
    lin_reg=LinearRegression()
   
    lin_reg.fit(X_train,y_train)
    endtime=datetime.datetime.now()
    print("endTime",endtime)
    timetake=str(endtime-starttime)
    print("timetake1",timetake)
    #print(timetake)
    t1 = timetake[2:4]
    t2 = str(t1)
    timetake=timetake[5:]
    print("timetake",(float(t2)*60 + float(timetake)))
    cost[1,n]=float(t2)*60 + float(timetake);
    y_pred=lin_reg.predict(X_test)
    endtime1=datetime.datetime.now()
    print("endTime1",endtime1)
    timetake1=str(endtime1-endtime)
    print("timetake11",timetake1)
    #print(timetake)
    t1 = timetake1[2:4]
    t2 = str(t1)
    timetake1=timetake1[5:]
    print("timetakepred",(float(t2)*60 + float(timetake1)))
    cost1=np.sum(np.power(y_pred-y_test,2))/(2*y_test.size)
    print (i," ",cost1)
       
    print("train_size ",(i/10)*100," ",result)
    cost[2,n]=float(t2)*60 + float(timetake1);
    n=n+1;
    



fig, ax = plt.subplots(figsize=(12, 8))
#ax.bar(index, avgCost)
np.savetxt("AverageCostOver50IterationsWithRegressionSLDTime.csv", cost, delimiter=',')
ax.plot(cost[0,:], cost[1,:], 'r', label='TrainingTime')
ax.plot(cost[0,:], cost[2,:], 'b', label='TestingTime')
#plt.xticks(index, cost[0,:], fontsize=5, rotation=30)
ax.set_xlabel('Amount of Data')
ax.set_ylabel('Time Taken')
ax.legend(loc=2)
fig.savefig('time1.png')# -*- coding: utf-8 -*-

