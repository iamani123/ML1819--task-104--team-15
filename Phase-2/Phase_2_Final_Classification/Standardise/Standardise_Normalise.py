import numpy as np

newX=np.zeros((20000,10))
X=np.genfromtxt('Letter_Recognition.csv', delimiter = ',',usecols=(1,2,3,4,5,6,7,8,9,10))

#min1=np.min(X, axis=0)
##max1=np.max(X, axis=0)
#for i in range(max1.size):
#    if((max1[i] - min1[i]) != 0) :
#        newX[:,i]=(X[:,i] - min1[i])/(max1[i] - min1[i])
#    else :
#        newX[:,i]=X[:,i]
               
mean1=np.mean(X, axis=0)
print(mean1)
std1=np.std(X, axis=0)
print(std1)
for i in range(10):
    if((std1[i]) != 0) :
        newX[:,i]=(X[:,i] - mean1[i])/(std1[i])
    else :
        newX[:,i]=X[:,i]
        
print(newX.mean(axis=0))
np.savetxt("Letter_recognition_standardise.csv", newX, delimiter=",")