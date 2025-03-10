# import require library 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv(r"C:\Users\navee\DS class files\ML class files\27th,28th,30th dec- slr with streamlit\27th,28th,30th- slr with streamlit\Salary_Data.csv")


# split the data to independent variable 
x = dataset.iloc[:,:-1]
# split the data to dependent variabel 
y = dataset.iloc[:, -1]


# as d.v is continus that regression algorithm 
# as in the data set we have 2 attribute we slr algo

# split the dataset to 80-20%

from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20,random_state=0)


#we called simple linear regression algoriytm from sklearm framework 

x_train = x_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1,1)


#fron  model builiding pipeline

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# test the model & create a predicted table 
y_pred = regressor.predict(x_test)


# visualize train data point ( 24 data)
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()


# visulaize test data point 
plt.scatter(x_test, y_test, color ="red")
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# slope is generrated from linear regress algorith which fit to dataset 
m = regressor.coef_

# interceppt also generatre by model.
c = regressor.intercept_


# predict or forcast the future the data which we not trained before 
y_12 = 9312 * 12 + 26780

Y_20 = 9312 *20 + 26780

# to check overfitting  ( low bias high variance)
bias = regressor.score(x_test,y_test)
bias 

# to check underfitting (high bias low variance)
variance = regressor.score(x_test,y_test)
variance 


# deployment in flask & html 
# mlops (azur, googlcolab, heroku, kubarnate)

import pickle


# Save the trained model to disk
filename = 'linear_regression_model.plk'


# Open a file in write-binary mode and dump the model
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
    
print("Model has been pickled and saved as linear_regression_model.pkl")
