import os
import pandas as pd
import numpy as np

os.getcwd()
os.chdir("D:\\Data Science\\Algorithmica\\Titanic") # use \\ for windows/linux compatibility

titanic_train = pd.read_csv("train.csv")
titanic_train.shape

titanic_train.info()
titanic_train.dtypes

# show sample data
titanic_train.head()
titanic_train.tail()

# provide summary statistics
titanic_train.describe()

# access the frame content by column
titanic_train["Age"]
titanic_train.Age
titanic_train[["PassengerId","Fare","Age"]]

# dropping a column: 0 -> column statistics; 1 -> row statistics
titanic_train1 = titanic_train.drop('Fare',1)
titanic_train1.shape

# slicing rows of frame (there is no , comma separator in Python)
titanic_train[0:4] # first 4 rows
titanic_train[:5]  # first 5 rows
titanic_train[-1:] # access last row
titanic_train[-2:] # access last two rows
# titanic_train[[0,3]] # does not give rows; gives two columns instead

# slicing subset of rows and columns
titanic_train.iloc[0,0]
titanic_train.iloc[0:3,0:3] # first 3 rows, 3 columns
titanic_train.iloc[[0,2],:] # 1st and 3rd row, all columns
titanic_train.iloc[:,0] # all rows, 1st column
titanic_train.iloc[:,[0,2]] # all rows, 1st and 3rd columns
titanic_train.loc[0:4,["Age"]] # label-based column; in .loc 0:4 includes 0 to 4 = 5 rows (and not to 3)

# slicing rows based on conditions
titanic_train[titanic_train.Age > 70]

# setting a column as index column
titanic_train.set_index('PassengerId') # set the index column
titanic_train.set_index('PassengerId', inplace=True)

# resetting index column
titanic_train.reset_index()
titanic_train.reset_index(inplace=True)