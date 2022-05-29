import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 

col_names = ['RID' , 'age', 'income', 'student', 'credit_rating', 'class_buy_computer'] 

# load dataset 

data = pd.read_csv("decisiontree2.csv") 

data.head() 

#split dataset in features and target variable 

feature_cols = ['age', 'income', 'credit_rating'] 

X = data[feature_cols] # Features 

y = data.class_buy_computer# Target variable 

# Split dataset into training set and test set 

X_train, X_test, y_train, y_test = train_test_split(X, y,  

test_size=0.3, random_state=1) # 70% training and 30% test 

# Create Decision Tree classifier object 

clf = DecisionTreeClassifier(criterion="entropy", 

max_depth=3) 

# Train Decision Tree Classifier 

clf = clf.fit(X_train,y_train) 

#Predict the response for test dataset 

y_pred = clf.predict(X_test) 

from sklearn.tree import export_graphviz 

from six import StringIO   

from IPython.display import Image   

import pydotplus 

dot_data = StringIO() 

export_graphviz(clf, out_file=dot_data,filled=True,  

rounded=True,special_characters=True, 

feature_names = feature_cols,class_names=['0','1']) 

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())   

graph.write_png('computer.png') 

Image(graph.create_png())
