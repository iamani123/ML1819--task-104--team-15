import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Letter_Recognition.csv")
X = data.iloc[:,1:17]  #independent columns
y = data.iloc[:,0]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(16).plot(kind='barh')
plt.show()

