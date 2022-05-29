
# import the modules needed for Multiple Linear Regression 
import pandas as pd # panel data
import matplotlib.pyplot as plt #mathematical plot 
from sklearn import linear_model #scikit learn 

# create the excel file and stored in CSV format.
data=pd.read_csv('m.csv') # read the CSV file
x=data[['x1','x2']].values # values of x1,x2 column
y=data['y'].values # vlaues of y column

regr=linear_model.LinearRegression() #LinearRegression
regr.fit(x,y) #fit x and y
newx1=int(input('enter the value of new x1  ')) #input of x1
newx2=int(input('enter the value of new x2  ')) #input of x2
newy=regr.predict([[newx1,newx2]]) #predicted value
print(newy) #print the output
df=pd.DataFrame(data,columns=["x1","x2","y"]) #converting into DataFrame
df.plot(kind="bar",figsize=(10,8)) #ploting
plt.show() #show the plot
