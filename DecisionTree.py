import numpy as np
import datetime
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
X=np.genfromtxt('credit_card.csv', delimiter = ',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23))
y=np.genfromtxt('credit_card.csv', delimiter = ',', dtype='str',usecols=(0))
acc =np.zeros((3,60))
n=0
for i in range(500,30001,500):
    starttime=datetime.datetime.now()
    print("startTime",starttime)
    idx = np.random.choice(range(30000), i, replace=False)
    new_X=X[idx,:]
    new_y=y[idx]
    acc[0,n]=i
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=.2, random_state=0)
    classifier = tree.DecisionTreeClassifier(max_depth=50)
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
fig, ax = plt.subplots(figsize=(12, 8))
#ax.bar(index, avgCost)
ax.plot(acc[0,:], acc[1,:], 'b', label='AccuracyForDecisionTree')

#plt.xticks(index, cost[0,:], fontsize=5, rotation=30)
ax.set_xlabel('Number Of Data', fontsize=5)
ax.set_ylabel('Accuracy', fontsize=5)
ax.legend(loc=2)
fig.savefig('AccuracyForDecisionTree.png')

fig1, ax1 = plt.subplots(figsize=(12, 8))
#ax.bar(index, avgCost)
ax1.plot(acc[0,:], acc[2,:], 'b', label='TimeforDecisionTree')

#plt.xticks(index, cost[0,:], fontsize=5, rotation=30)
ax1.set_xlabel('Number Of Data', fontsize=5)
ax1.set_ylabel('TimeTaken', fontsize=5)
ax1.legend(loc=2)
fig1.savefig('GraphForDecisionTree.png')
