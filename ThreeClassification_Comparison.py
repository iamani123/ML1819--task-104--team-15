import numpy as np
import datetime
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
X=np.genfromtxt('DataSetLetter.csv', delimiter = ',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
y=np.genfromtxt('DataSetLetter.csv', delimiter = ',', dtype='str',usecols=(0))
acc =np.zeros((7,40))
from sklearn.svm import SVC
n=0
for i in range(500,20001,500):
    starttime=datetime.datetime.now()
    print("startTime",starttime)
    idx = np.random.choice(range(20000), i, replace=False)
    new_X=X[idx,:]
    new_y=y[idx]
    acc[0,n]=i
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=.2, random_state=0)
    #classifier=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
    classifier=LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    score=0
    
    for j in range(len(y_test)):
        if(y_test[j]==y_pred[j]):
            score=score+1
    #print(score,"/",len(y_test))
    result=(score/len(y_test))*100       
    print(i," ",result)
    acc[1,n]=result;
   
    endtime=datetime.datetime.now()
    print("endTime",endtime)
    timetake=str(endtime-starttime)
    timetake=timetake[5:]
    print("timetake",timetake)
    acc[2,n]=float(timetake)
    print("")
    
    n=n+1
n=0
for i in range(500,20001,500):
    starttime=datetime.datetime.now()
    print("startTime",starttime)
    idx = np.random.choice(range(20000), i, replace=False)
    new_X=X[idx,:]
    new_y=y[idx]
    
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=.2, random_state=0)
    classifier1=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
    #classifier=LogisticRegression(random_state=0)
    classifier1.fit(X_train,y_train)
    y_pred=classifier1.predict(X_test)
    score=0
    
    for j in range(len(y_test)):
        if(y_test[j]==y_pred[j]):
            score=score+1
    #print(score,"/",len(y_test))
    result=(score/len(y_test))*100       
    print(i," ",result)
    acc[3,n]=result;
   
    endtime=datetime.datetime.now()
    print("endTime",endtime)
    timetake=str(endtime-starttime)
    timetake=timetake[5:]
    print("timetake",timetake)
    acc[4,n]=float(timetake)
    print("")
    
    n=n+1 
n=0
clf=SVC(gamma='auto')
for i in range(500,20001,500):
    starttime=datetime.datetime.now()
    print("startTime",starttime)
    idx = np.random.choice(range(20000), i, replace=False)
    new_X=X[idx,:]
    new_y=y[idx]
    
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=.2, random_state=0)
    #classifier1=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
    #classifier=LogisticRegression(random_state=0)
    #classifier1.fit(X_train,y_train)
    #y_pred=classifier1.predict(X_test)
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    y_pred=svclassifier.predict(X_test)
    score=0
    
    for j in range(len(y_test)):
        if(y_test[j]==y_pred[j]):
            score=score+1
    #print(score,"/",len(y_test))
    result=(score/len(y_test))*100       
    print(i," ",result)
    acc[5,n]=result;
   
    endtime=datetime.datetime.now()
    print("endTime",endtime)
    timetake=str(endtime-starttime)
    timetake=timetake[5:]
    print("timetake",timetake)
    acc[6,n]=float(timetake)
    print("")    
    n=n+1    
fig, ax = plt.subplots(figsize=(12, 8))
#ax.bar(index, avgCost)
ax.plot(acc[0,:], acc[1,:], 'b', label='AccForLogistic')
ax.plot(acc[0,:], acc[3,:], 'r', label='AccForKNN')
ax.plot(acc[0,:], acc[5,:], 'g', label='AccForSVM')
#plt.xticks(index, cost[0,:], fontsize=5, rotation=30)
ax.set_xlabel('Number Of Data', fontsize=5)
ax.set_ylabel('Accuracy', fontsize=5)
ax.legend(loc=2)
fig.savefig('AccForLogAlgo.png')

fig1, ax1 = plt.subplots(figsize=(12, 8))
#ax.bar(index, avgCost)
ax1.plot(acc[0,:], acc[2,:], 'b', label='timeForLogistic')
ax1.plot(acc[0,:], acc[4,:], 'r', label='timeForKNN')
ax1.plot(acc[0,:], acc[6,:], 'g', label='timeForSVM')
#plt.xticks(index, cost[0,:], fontsize=5, rotation=30)
ax1.set_xlabel('Number Of Data', fontsize=5)
ax1.set_ylabel('TimeTaken', fontsize=5)
ax1.legend(loc=2)
fig1.savefig('AccForlogAlgoTime.png')
