# Import required packages
from tensorflow.contrib import learn
from tensorflow.contrib import layers
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import seaborn as sb # for graphics
import matplotlib.pyplot as plt

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("/home/algo/Algorithmica/tensorflow-builtin-estimators")
os.getcwd()

# reading directly using tensorflow's api
# train2.csv does not have headers. Instead, first row has #rows, #columns
sample = learn.datasets.base.load_csv_with_header(
        filename="train2.csv",
        target_dtype=np.int,
        features_dtype=np.float32, target_column=-1) # '-1' means last column

type(sample)
sample.data
sample.data.shape
type(sample.data)
sample.target
sample.target.shape
type(sample.target)

#feature_columns argument expects list of tensorflow feature types
feature_cols = [layers.real_valued_column("", dimension=2)]

# If n_classes > 2, it is multi-class classification. Although we are trying to learn about a 
# single perceptron, LinearClassifier internally uses a layer of perceptrons to classify.
classifier = learn.LinearClassifier(feature_columns=feature_cols,
                                            n_classes=2, # binary classificationi
                                            model_dir="/home/algo/Algorithmica/tmp",
                                            enable_centered_bias=False)
# By default, enable_centered_bias is True

# If enable_centered_bias = False, linear classifier equation is:
#   f(x,y) = w_1*x  + w_2*y + bias = 0
#   To predict, f(x,y) < 0 ==> [x,y] is Class 0, else [x,y] is Class 1

# If enable_centered_bias = True, linear classifier equation is:
#   f(x,y) = w_1*x  + w_2*y + bias + centered_bias_weight = 0
#   To predict, f(x,y) < 0 ==> [x,y] is Class 0, else [x,y] is Class 1

# Note that enabling/disabling the centered bias can result in different predictions for "border-case" points

classifier.fit(x=sample.data, y=sample.target, steps=1000)

#access the learned model parameters
classifier.weights_
classifier.bias_

# valid only when enable_centered_bias=True in learn.LinearClassifier()
# centered_bias_weight = classifier.get_variable_value("centered_bias_weight")

for var in classifier.get_variable_names():
    print var, " - ", classifier.get_variable_value(var)
    
# w1*x + w2*y + b = 0.
p1 = [0,-classifier.bias_[0]/classifier.weights_[1]] # (0, -b/w2)
p2 = [-classifier.bias_[0]/classifier.weights_[0],0] # (-b/w1, 0)

df = pd.DataFrame(data=np.c_[sample.data, sample.target.astype(int)], columns=['x1','x2','label'])
sb.swarmplot(x='x1', y='x2', data=df, hue='label', size=10)
plt.plot(p1, p2, 'b-', linewidth = 2)

# predict the outcome using model                                  
test = np.array([[60.4,21.5],[200.1,26.1],[50,62],[50,63],[70,37],[70,38]])
predictions = classifier.predict(test)
predictions # [0,1,0,1,0,1]

test[0,0]
test[0,1]

# Understanding how the predictions were made
# Since enable_centered_bias = False, linear classifier equation is:
#   f(x,y) = w_1*x  + w_2*y + bias = 0
#   To predict, f(x,y) < 0 ==> [x,y] is Class 0, else [x,y] is Class 1
test[0,0]*classifier.weights_[0]  + test[0,1]*classifier.weights_[1] + classifier.bias_ # -0.817 ==> class 0
test[1,0]*classifier.weights_[0]  + test[1,1]*classifier.weights_[1] + classifier.bias_ # 4.432 ==> class 1
test[2,0]*classifier.weights_[0]  + test[2,1]*classifier.weights_[1] + classifier.bias_ # -0.021 ==> class 0
test[3,0]*classifier.weights_[0]  + test[3,1]*classifier.weights_[1] + classifier.bias_ # 0.008 ==> class 1
test[4,0]*classifier.weights_[0]  + test[4,1]*classifier.weights_[1] + classifier.bias_ # -0.015 ==> class 0
test[5,0]*classifier.weights_[0]  + test[5,1]*classifier.weights_[1] + classifier.bias_ # 0.013 ==> class 1
    
# Predict the class of random points (when enable_centered_bias = False)
x = 70
y = 37
classifier.weights_[0]*x + classifier.weights_[1]*y + classifier.bias_
classifier.predict(np.array([[x,y]]))

# loop version for seeing the individual points and predictions in test data
for i in range(len(test)):
    value = test[i,0]*classifier.weights_[0]  + test[i,1]*classifier.weights_[1] + classifier.bias_
    print "Score: ", value, " ==> ", predictions[i]