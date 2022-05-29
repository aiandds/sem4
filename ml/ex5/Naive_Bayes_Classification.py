# packages included 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
# read the data set
data=pd.read_csv("dtree2.csv")
data.head()
col_names = ['RID', 'age', 'income', 'credit_rat', 'class_buy_computer']
# data preprocessing convert the string into integer
number = LabelEncoder()
data['age'] = number.fit_transform(data['age'])
data['income'] = number.fit_transform(data['income'])
data['student'] = number.fit_transform(data['student'])
data['credit_rat'] = number.fit_transform(data['credit_rat'])
data['class_buy_computer'] = number.fit_transform(data['class_buy_computer'])
#selected the important features
feature_cols = ['age', 'income', 'credit_rat']

X = data[feature_cols] # independent variable
y = data.class_buy_computer # dependent variable
# split the data set into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# generate the model using gaussian naive bayes
model = GaussianNB()
model.fit(X_train, y_train)

# apply the metrics to find accuracy, prediction and confusion matrix
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
cm=confusion_matrix(y,model.predict(X))
print(classification_report(y,model.predict(X)))

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

