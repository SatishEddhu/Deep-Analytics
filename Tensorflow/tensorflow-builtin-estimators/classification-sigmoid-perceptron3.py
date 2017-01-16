# Import required packages
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from sklearn import model_selection
import tensorflow as tf
import os
import pandas as pd
import numpy as np

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("/home/algo/Algorithmica/tensorflow-builtin-estimators")

# Read the train data
sample = pd.read_csv("train1.csv")
sample.shape
sample.info()

x = learn.extract_pandas_data(sample[['x1','x2']])
y = learn.extract_pandas_labels(sample[['label']])
# The above code is equivalent to the below commented python code
# But the above 'learn' version is recommended since it can run on any device - CPU, GPU,
# mobile, whereas the below python code will run only on CPU
#    x = sample[['x1','x2']].as_matrix()
#    y = sample[['label']].as_matrix()

# Divide the input data into train and validation
x_train,x_validate,y_train,y_validate = model_selection.train_test_split(x, y, test_size=0.2, random_state=100)
type(x_train)

#feature engineering
feature_cols = [layers.real_valued_column("", dimension=2)]

#build the model configuration              
classifier = learn.LinearClassifier(feature_columns=feature_cols,
                                            n_classes=2,
                                            model_dir="/home/algo/Algorithmica/tmp")

#build the model
classifier.fit(x=x_train, y=y_train, steps=1000)
classifier.weights_
classifier.bias_

# By default, enable_centered_bias = True in learn.LinearClassifier
centered_bias_weight = classifier.get_variable_value("centered_bias_weight")

#evaluate the model using validation set
results = classifier.evaluate(x=x_validate, y=y_validate, steps=1)
type(results)
for key in sorted(results):
    print "%s:%s" % (key, results[key])
    
# Predict the outcome of test data using model
# Note how we get different predictions on the same data with enable_centered_bias=True (compared to classification-sigmoid-perceptron1.py)
test = np.array([[60.4,21.5],[200.1,26.1],[50,62],[50,63],[70,37],[70,38]])
predictions = classifier.predict(test)
predictions # [0,1,0,0,1,1]

# Understanding how the predictions were made
# Since enable_centered_bias = True, linear classifier equation is:
#   f(x,y) = w_1*x  + w_2*y + bias + centered_bias_weight = 0
#   To predict, f(x,y) < 0 ==> [x,y] is Class 0, else [x,y] is Class 1
total_bias = centered_bias_weight + classifier.bias_
test[0,0]*classifier.weights_[0]  + test[0,1]*classifier.weights_[1] + total_bias # -0.668 ==> class 0
test[1,0]*classifier.weights_[0]  + test[1,1]*classifier.weights_[1] + total_bias # 4.552 ==> class 1
test[2,0]*classifier.weights_[0]  + test[2,1]*classifier.weights_[1] + total_bias # -0.047 ==> class 0
test[3,0]*classifier.weights_[0]  + test[3,1]*classifier.weights_[1] + total_bias # -0.022 ==> class 0
test[4,0]*classifier.weights_[0]  + test[4,1]*classifier.weights_[1] + total_bias # 0.065 ==> class 1
test[5,0]*classifier.weights_[0]  + test[5,1]*classifier.weights_[1] + total_bias # 0.090 ==> class 1
