# -*- coding: utf-8 

import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC

clf=SVC(gamma='auto')
acc =np.zeros((2,9))
n=0
for i in range(500,4602,500):
    data=np.genfromtxt('spambase.data',delimiter=',')
    print(data)
    idx = np.random.choice(range(4601), size=i)
    X=data[idx,0:56]
    y=data[idx,57]
    new_X=X
    new_y=y
    acc[0,n]=i
    np.any(np.isnan(new_X))
   # np.any(np.isnan(new_y))
    np.all(np.isfinite(new_X))
    #new_X[new_X.eq(np.inf).any(axis=1)]
    #np.all(np.isfinite(new_y))
    #new_X.as_matrix().astype(np.float)
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=0.20)
    print(new_X)
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
    acc[1,n]=result;
    n=n+1
fig, ax = plt.subplots(figsize=(12, 8))
#ax.bar(index, avgCost)
ax.plot(acc[0,:], acc[1,:], 'r', label='Cost')
#plt.xticks(index, cost[0,:], fontsize=5, rotation=30)
ax.set_xlabel('Number Of Data', fontsize=5)
ax.set_ylabel('Accuracy', fontsize=5)
ax.legend(loc=2)
fig.savefig('AccForSVMAlgo_Spam.png')
#from sklearn.metrics import classification_report, confusion_matrix  
#print(confusion_matrix(y_test,y_pred))  
#print(classification_report(y_test,y_pred))  
