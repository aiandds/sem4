# import the modules needed for linear regression
import numpy as np    # numerical python 
import pandas as pd    # panel data
import matplotlib.pyplot as plt  #mathematical plot 

# create the excel file and stored in CSV format.
data = pd.read_csv('cricket.csv')   # read the CSV file
x = data['over']                              # index of over column
y = data['run']                                # index of run column 
print(data.head())                        # print the first 5 rows of input data as default

# define the function linear regression to find the regression line
def simplelinear(x, y):     
    xmean = x.mean()     # find the mean value of x
    ymean = y.mean()      # find the mean value of y
    sp = ((x - xmean) * (y - ymean)).sum()   # sum of product of x and y
    sd = ((x - xmean)**2).sum() # sum of standard deviation
    m = sp / sd  # find the m value
    c = ymean - (m*xmean)  # find the c value
    regline = 'y = {}x + {}'.format(round(m, 3),c)
    return (m, c, regline)

m, c, regline = simplelinear(x, y)
print('Regression Line: ', regline)

# define the function to find the correlation between x and y
def corrcoef(x, y):
    N = len(x)
    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    return R
#call the function corrcoef
R = corrcoef(x, y)
print('Correlation Coef.: ', R)
print('"Goodness of Fit": ', R**2)

#define the function to predict the new value
def predict(m, c, newx):
    y = m * newx + c
    return y

#call the function predict
newx=int(input("Enter the new x value to predict"))
print("Result=",predict(m,c,newx))

#visualize the data using matplot lib
plt.figure(figsize=(12,5))    # set the figure size
plt.scatter(x, y) # scatter plot
plt.title('How Over Affects Runs') # title of the figure
plt.xlabel('Over', fontsize=15) # x label 
plt.ylabel('Runs', fontsize=15)   # y label
plt.plot(x, m*x + c)  # plot the regression line
plt.show()   # print the plotted line
