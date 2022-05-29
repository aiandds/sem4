#import dataset
from sklearn import datasets
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


#load dataset from scikit 
first=pd.read_csv("tshirt.csv")

shirt = pd.DataFrame(first) 

number = LabelEncoder()
shirt['Tshirt_size'] = number.fit_transform(shirt['Tshirt_size'])

y = shirt['Tshirt_size']
x = shirt.drop(columns=['Tshirt_size'])

#70% train,30% test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

cf=svm.SVC(kernel='linear')
cf.fit(x_train,y_train)
y_pred=cf.predict(x_test)

# print output
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
print("Precision:",metrics.precision_score(y_test,y_pred))
print("Recall:",metrics.recall_score(y_test,y_pred))

