# packages included 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
import time

# load dataset
start = time.time()
data=pd.read_csv("tshirt.csv")
print(data.head())
number = LabelEncoder()
data['Tshirt_size'] = number.fit_transform(data['Tshirt_size'])
feature_cols = ['Height','Weight']

#split dataset in features and target variable
X = data[feature_cols]
y = data.Tshirt_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
knn=KNeighborsClassifier(n_neighbors=1) #KNN
knn.fit(X_train,y_train) #Knn fit
pred=knn.predict(X_test)
accuracy = accuracy_score(y_test, pred) #accuracy
cm=confusion_matrix(y,knn.predict(X))

#visulaization of heatmap
fig,ax=plt.subplots(figsize=(8,8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1),ticklabels=('Predicted 0s','Predicted 1s'))
ax.yaxis.set(ticks=(0,1),ticklabels=('Actual 0s','Actual 1s'))
ax.set_ylim(1.5,-0.5)
for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i,j],ha='center',va='center',color='red')
plt.show()

print(classification_report(y,knn.predict(X))) #print the result

end = time.time()

speed = end-start

print("speed = ",speed)


