import numpy as np
import datetime
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
X=np.genfromtxt('updated_creditCard_preprocessing.csv', delimiter = ',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23))
y=np.genfromtxt('updated_creditCard_preprocessing.csv', delimiter = ',',usecols=(0))
acc =np.zeros((11,60))

n=0
for i in range(500,30001,500):
    starttime=datetime.datetime.now()
    print("startTime",starttime)
    idx = np.random.choice(range(30000), i, replace=False)
    new_X=X[idx,:]
    new_y=y[idx]
    acc[0,n]=i
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=.2, random_state=0)
    #classifier=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
    classifier=LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    score=0
    TP=0
    TN=0
    FN=0
    FP=0
    for j in range(len(y_test)):
       # print("y_test[j],y_pred[j]",y_test[j]," ",y_pred[j])
        if(y_test[j]==y_pred[j] and y_test[j]==1.0):
        #    print("TP")
            TP=TP+1
        if(y_test[j]==y_pred[j] and y_test[j]==0.0):
         #   print("TN")
            TN=TN+1
        if(y_test[j]!=y_pred[j] and y_test[j]==1.0):
          #  print("FN")
            FN=FN+1
        if(y_test[j]!=y_pred[j] and y_test[j]==0.0):
           # print("FP")
            FP=FP+1
    print("TP,TN,FP,FN",TP," ",TN," ",FP," ",FN)
    if((TP+FP)!=0):
        precision=TP/(TP+FP)
    else:
        precision=0
    print("precision ",precision)
    if((TP+FN)!=0):
        recall=TP/(TP+FN)
    else:
        recall=0
    
    print("recall ",recall)
    if((recall+precision)!=0):
        result=(2*recall*precision)/(recall+precision)
    else:
        result=0
    
    #print(score,"/",len(y_test))
    #result=(score/len(y_test))*100       
    print(i," ",result)
    acc[1,n]=result;
   
    endtime=datetime.datetime.now()
    print("endTime",endtime)
    timetake=str(endtime-starttime)
    #print(timetake)
    t1 = timetake[2:4]
    t2 = str(t1)
    timetake=timetake[5:]
    print("timetake",(float(t2)*60 + float(timetake)))
    acc[2,n]=(float(t2)*60 + float(timetake))
    print("")
    
    n=n+1
n=0
for i in range(500,30001,500):
    starttime=datetime.datetime.now()
    print("startTime",starttime)
    idx = np.random.choice(range(30000), i, replace=False)
    new_X=X[idx,:]
    new_y=y[idx]
    
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=.2, random_state=0)
    classifier1=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
    #classifier=LogisticRegression(random_state=0)
    classifier1.fit(X_train,y_train)
    y_pred=classifier1.predict(X_test)
    score=0
    TP=0
    TN=0
    FN=0
    FP=0
    for j in range(len(y_test)):
        if(y_test[j]==y_pred[j] and y_pred[j]==1):
            TP=TP+1
        if(y_test[j]==y_pred[j] and y_pred[j]==0):
            TN=TN+1
        if(y_test[j]!=y_pred[j] and y_pred[j]==1):
            FP=FP+1
        if(y_test[j]!=y_pred[j] and y_pred[j]==0):
            FN=FN+1
    print("TP,TN,FP,FN",TP," ",TN," ",FP," ",FN)
    if((TP+FP)!=0):
        precision=TP/(TP+FP)
    else:
        precision=0
    print("precision ",precision)
    if((TP+FN)!=0):
        recall=TP/(TP+FN)
    else:
        recall=0
    if((recall+precision)!=0):
        result=(2*recall*precision)/(recall+precision)
    else:
        result=0
    print(i," ",result)
    acc[3,n]=result;
   
    endtime=datetime.datetime.now()
    print("endTime",endtime)
    timetake=str(endtime-starttime)
    t1 = timetake[2:4]
    t2 = str(t1)
    timetake=timetake[5:]
    print("timetake",(float(t2)*60 + float(timetake)))
    acc[4,n]=(float(t2)*60 + float(timetake))
    print("")
    
    n=n+1 
n=0
clf=SVC(gamma='auto')
for i in range(500,30001,500):
    starttime=datetime.datetime.now()
    print("startTime",starttime)
    idx = np.random.choice(range(30000), i, replace=False)
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
    
    TP=0
    TN=0
    FN=0
    FP=0
    for j in range(len(y_test)):
        if(y_test[j]==y_pred[j] and y_pred[j]==1):
            TP=TP+1
        if(y_test[j]==y_pred[j] and y_pred[j]==0):
            TN=TN+1
        if(y_test[j]!=y_pred[j] and y_pred[j]==1):
            FP=FP+1
        if(y_test[j]!=y_pred[j] and y_pred[j]==0):
            FN=FN+1
    print("TP,TN,FP,FN",TP," ",TN," ",FP," ",FN)
    if((TP+FP)!=0):
        precision=TP/(TP+FP)
    else:
        precision=0
    print("precision ",precision)
    if((TP+FN)!=0):
        recall=TP/(TP+FN)
    else:
        recall=0
    if((recall+precision)!=0):
        result=(2*recall*precision)/(recall+precision)
    else:
        result=0  
    print(i," ",result)
    acc[5,n]=result;

    endtime=datetime.datetime.now()
    print("endTime",endtime)
    timetake=str(endtime-starttime)
    t1 = timetake[2:4]
    t2 = str(t1)
    timetake=timetake[5:]
    print("timetake",(float(t2)*60 + float(timetake)))
    acc[6,n]=(float(t2)*60 + float(timetake))
    print("")
    
    n=n+1    
n=0
for i in range(500,30001,500):
    starttime=datetime.datetime.now()
    print("startTime",starttime)
    idx = np.random.choice(range(30000), i, replace=False)
    new_X=X[idx,:]
    new_y=y[idx]
   
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=.2, random_state=0)
    #classifier=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
    #classifier=LogisticRegression(random_state=0)
   # classifier = GaussianNB()
    classifier = tree.DecisionTreeClassifier(max_depth=50)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    score=0
    TP=0
    TN=0
    FN=0
    FP=0
    for j in range(len(y_test)):
        if(y_test[j]==y_pred[j] and y_pred[j]==1):
            TP=TP+1
        if(y_test[j]==y_pred[j] and y_pred[j]==0):
            TN=TN+1
        if(y_test[j]!=y_pred[j] and y_pred[j]==1):
            FP=FP+1
        if(y_test[j]!=y_pred[j] and y_pred[j]==0):
            FN=FN+1
   
    if((recall+precision)!=0):
        result=(2*recall*precision)/(recall+precision)
    else:
        result=0 
    print("TP,TN,FP,FN",TP," ",TN," ",FP," ",FN)
    if((TP+FP)!=0):
        precision=TP/(TP+FP)
    else:
        precision=0
    print("precision ",precision)
    if((TP+FN)!=0):
        recall=TP/(TP+FN)
    else:
        recall=0
    print("recall ",recall)
    print(i," ",result)
    acc[7,n]=result;
   
    endtime=datetime.datetime.now()
    print("endTime",endtime)
    timetake=str(endtime-starttime)
    t1 = timetake[2:4]
    t2 = str(t1)
    timetake=timetake[5:]
    print("timetake",(float(t2)*60 + float(timetake)))
    acc[8,n]=(float(t2)*60 + float(timetake))
    print("")
    
    n=n+1
n=0
for i in range(500,30001,500):
    starttime=datetime.datetime.now()
    print("startTime",starttime)
    idx = np.random.choice(range(30000), i, replace=False)
    new_X=X[idx,:]
    new_y=y[idx]
   
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=.2, random_state=0)
    #classifier=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
    #classifier=LogisticRegression(random_state=0)
   # classifier = GaussianNB()
    #classifier = tree.DecisionTreeClassifier(max_depth=50)
    classifier = RandomForestClassifier()
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    score=0
    TP=0
    TN=0
    FN=0
    FP=0
    for j in range(len(y_test)):
        if(y_test[j]==y_pred[j] and y_pred[j]==1):
            TP=TP+1
        if(y_test[j]==y_pred[j] and y_pred[j]==0):
            TN=TN+1
        if(y_test[j]!=y_pred[j] and y_pred[j]==1):
            FP=FP+1
        if(y_test[j]!=y_pred[j] and y_pred[j]==0):
            FN=FN+1
    if((TP+FP)!=0):
        precision=TP/(TP+FP)
    else:
        precision=0
    print("precision ",precision)
    if((TP+FN)!=0):
        recall=TP/(TP+FN)
    else:
        recall=0
    print("recall ",recall)
    if((recall+precision)!=0):
        result=(2*recall*precision)/(recall+precision)
    else:
        result=0
    acc[9,n]=result;
   
    endtime=datetime.datetime.now()
    print("endTime",endtime)
    timetake=str(endtime-starttime)
    t1 = timetake[2:4]
    t2 = str(t1)
    timetake=timetake[5:]
    print("timetake",(float(t2)*60 + float(timetake)))
    acc[10,n]=(float(t2)*60 + float(timetake))
    print("")
    
    n=n+1

np.savetxt("accuracy_DataSetLetter.csv", acc, delimiter=',')
fig, ax = plt.subplots(figsize=(12, 8))
#ax.bar(index, avgCost)
ax.plot(acc[0,:], acc[1,:], 'b', label='Logistic')
ax.plot(acc[0,:], acc[3,:], 'r', label='KNN')
ax.plot(acc[0,:], acc[5,:], 'g', label='SVM')
ax.plot(acc[0,:], acc[7,:], 'm', label='DecisionTree')
ax.plot(acc[0,:], acc[9,:], 'k', label='RandomForest')
#plt.xticks(index, cost[0,:], fontsize=5, rotation=30)
ax.set_xlabel('Number Of Data')
ax.set_ylabel('Accuracy')
ax.legend(loc=2)
fig.savefig('AccForAllClassification_DataSet.png')

fig1, ax1 = plt.subplots(figsize=(12, 8))
#ax.bar(index, avgCost)
ax1.plot(acc[0,:], acc[2,:], 'b', label='Logistic')
ax1.plot(acc[0,:], acc[4,:], 'r', label='KNN')
ax1.plot(acc[0,:], acc[6,:], 'g', label='SVM')
ax1.plot(acc[0,:], acc[8,:], 'm', label='DecisionTree')
ax1.plot(acc[0,:], acc[10,:], 'k', label='RandomForest')
#plt.xticks(index, cost[0,:], fontsize=5, rotation=30)
ax1.set_xlabel('Number Of Data', fontsize=5)
ax1.set_ylabel('TimeTaken', fontsize=5)
ax1.legend(loc=2)
fig1.savefig('AccForAllClassification_Dataset.png')