#import dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("dtree2.csv")
col_names = ['RID', 'age', 'income', 'credit_rat', 'class_buy_computer']

number = LabelEncoder()
data['age'] = number.fit_transform(data['age'])
data['income'] = number.fit_transform(data['income'])
data['student'] = number.fit_transform(data['student'])
data['credit_rat'] = number.fit_transform(data['credit_rat'])
data['class_buy_computer'] = number.fit_transform(data['class_buy_computer'])
#selected the important features
feature_cols = ['age','income','credit_rat']


km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(data[feature_cols])
#print(y_predicted)
data['cluster'] = y_predicted

#print(data)

df1 = data[data.cluster==0]
df2 = data[data.cluster==1]


#print(df1)
#print(df2)

print(data)


x = data.drop(['cluster','class_buy_computer'],axis=1)
y = data['cluster']

#print("THE VALUE OF x \n",x)

scal = StandardScaler()
one = scal.fit_transform(x)

#pca
pca = PCA(n_components=2)
fnl_pca = pca.fit_transform(one)
print(fnl_pca)
#plotting
plt.figure(figsize=(8,6))
plt.scatter(fnl_pca[:,0],fnl_pca[:,1],c=y)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("Principal Component Analysis")
plt.show()
