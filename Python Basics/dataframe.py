import os
import pandas as pd
import random as r # can also use numpy for random() function
import numpy as np

os.getcwd()
os.chdir("D:\\Deep Analytics\\DigitRecognizer") # use \\ for windows/linux compatibility

digit_train = pd.read_csv("train.csv")
type(digit_train) # pandas.core.frame.DataFrame
digit_train.shape # dimensions
digit_train.info() # summary info on data frame
digit_train.dtypes # type of every column

# Data exploration
digit_train.head()
digit_train.tail()
digit_train.head(10)

# equivalent to summary(train) in R
digit_train.describe()
# Provides details like mean, min,max,sd, quratiles for each column
# Thus gives 'central tendency' and 'spread'
# Shows stats for 'label' as it is also considered a numerical type

digit_train.label # equivalent to train$label in R
digit_train.label = digit_train.label.astype('category') # casting to categorical type
digit_train.dtypes # label type shown as 'category' instead of int64

digit_train.describe()
# No stats for categorical columns. Perhaps because Python was adapted later for
# data analytics, and so categorical types came up very late in the language

# accessing multiple columns by index or name
digit_train[[0,1]] # we don't have 0:1 syntax like in R; so we use a list [0,1] instead
digit_train[['label', 'pixel0']]

# extracting one column by name or index
digit_train[[0]]
digit_train[['label']]

# access rows based on condition
digit_train[digit_train.pixel200 > 100]

# read the test data
digit_test = pd.read_csv("test.csv")
digit_test.shape

# randomly predict the labels
digit_test.label = r.randrange(10) # [0,9]
# Vectorization NOT by default on basic data types.
# Vectorization only in np.array from numpy package

digit_train1 = digit_train[[0,1]]
digit_train1.to_csv("submission.csv")
digit_train1.to_csv("submission.csv", index_label=["label"])

# To create 'Image Id" 
range(1,28001,1)
