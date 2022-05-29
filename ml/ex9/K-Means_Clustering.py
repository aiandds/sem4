
#import dataset
import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("tshirt.csv")
df= pd.DataFrame(data) 


plt.xlabel('Height')
plt.ylabel('Weight')
plt.scatter(df['Height'], df['Weight'],color="green")
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Height','Weight']])
#print(y_predicted)
df['cluster'] = y_predicted



df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]


plt.scatter(df1['Height'], df1['Weight'],color="green",label='cluster 1')
plt.scatter(df2['Height'],df2['Weight'],color='red',label='cluster 2')
plt.scatter(df3['Height'],df3['Weight'],color='blue',label='cluster 3')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='tomato',marker='*',label='centroid')
plt.legend()
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()





x = df.drop(['Tshirt_size'], axis=1)
print(x.info())
y = km.fit_predict(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=109)
km.fit(x_train,y_train)
pre = km.predict(x_test)


print("Accuracy:",metrics.accuracy_score(pre,y_test))
