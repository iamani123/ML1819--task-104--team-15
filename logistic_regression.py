import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
X=np.genfromtxt('credit_card.csv', delimiter = ',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23))
y=np.genfromtxt('credit_card.csv', delimiter = ',',usecols=(0))
print(X)
print(y )
acc =np.zeros((2,60))
n=0
for i in range(500,30001,500):
    idx = np.random.choice(range(30000), i, replace=False)
    #np.random.sample(range(30001), i)
    print(idx)
    #np.random.random_integers(20,size=(i)) 
    #np.random.randint(30000, size=i)
    new_X=X[idx,:]
    new_y=y[idx]
    acc[0,n]=i
    X_train, X_test, y_train,y_test  = train_test_split(new_X,new_y,test_size=.2, random_state=0)
    classifier=LogisticRegression()
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