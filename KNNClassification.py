import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
X=np.genfromtxt('DataSetLetter.csv', delimiter = ',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
y=np.genfromtxt('DataSetLetter.csv', delimiter = ',', dtype='str',usecols=(0))
acc =np.zeros((2,40))
n=0
for i in range(500,20001,500):
    idx = np.random.randint(20000, size=i)
    new_X=X[idx,:]
    new_y=y[idx]
    acc[0,n]=i
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=.2, random_state=0)
    classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
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
    n=n+1
fig, ax = plt.subplots(figsize=(12, 8))
#ax.bar(index, avgCost)
ax.plot(acc[0,:], acc[1,:], 'r', label='Cost')
#plt.xticks(index, cost[0,:], fontsize=5, rotation=30)
ax.set_xlabel('Number Of Data', fontsize=5)
ax.set_ylabel('Accuracy', fontsize=5)
ax.legend(loc=2)
fig.savefig('AccForKNNAlgo.png')