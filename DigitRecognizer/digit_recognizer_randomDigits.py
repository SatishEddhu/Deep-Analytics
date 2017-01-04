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


# randomly predict the labels
imageid = range(1,28001,1)
len(imageid)
label = np.random.randint(0,10,28000)
len(label)
dict = {'ImageId':imageid, 'Label': label}
out_df = pd.DataFrame.from_dict(dict)
out_df.shape
out_df.head(5)
out_df.set_index('ImageId', inplace=True) # If inplace is False, another column ImageId would have been added

out_df.to_csv("submission.csv")
