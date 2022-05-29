#import dataset
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


df = pd.read_csv('tshirt_size.csv')

model=IsolationForest(n_estimators=1000,max_samples='auto',contamination=float(0.2),max_features=1.0)
print(model.fit(df[['Height']]))

df['scores']=model.decision_function(df[['Height']])
df['anomaly']=model.predict(df[['Height']])
print(df)

df1= df[df.anomaly != 1]

#plot
plt.scatter(df['Height'], df['Weight'],color='tomato',marker='*')
plt.scatter(df1['Height'],df1['Weight'],color='green',label='Anomaly')
plt.legend()
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()
