import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("D:\\Subjects\\ir\\ML1819--task-104--team-15\\Phase-2\\updated_letter_dataset_standardise.csv")
X = data.iloc[:,1:16]  #independent columns
y = data.iloc[:,0]    #target column i.e price range
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(16,16))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")