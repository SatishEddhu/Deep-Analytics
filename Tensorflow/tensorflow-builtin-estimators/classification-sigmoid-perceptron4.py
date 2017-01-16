# Import required packages
from tensorflow.contrib import learn
from tensorflow.contrib import layers
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("/home/algo/Algorithmica/tensorflow-builtin-estimators")
os.getcwd()

# Read the train data
# hascolumns x1(float), x2(float), label(0/1)
sample = pd.read_csv("train1.csv")
sample.shape
sample.info()

FEATURES = ['x1','x2']
LABEL = ['label']

# function must return tensors
# input function must return featurecols and labels separately   
def input_function(dataset, train=False):
    feature_cols = {k : tf.constant(dataset[k].values) 
                        for k in FEATURES}
    if train:
        labels = tf.constant(dataset[LABEL].values)
        return feature_cols, labels
    return feature_cols
    
# Build the model with right feature tranformation
feature_cols = [layers.real_valued_column(k) for k in FEATURES]

classifier = learn.LinearClassifier(feature_columns=feature_cols,
                                    n_classes=2,
                                    model_dir='/home/algo/Algorithmica/tmp')

classifier.fit(input_fn = lambda: input_function(sample,True), steps=1000)

# Weights and bias learned are different from that in classification-sigmoid-perceptron3.py although both have
# the same default value for enable_centered_bias. Why???
classifier.weights_ 
classifier.bias_ 

# Predict the outcome using model
dict = {'x1':[60.4,200.1,50,50,70,70], 'x2':[21.5,26.1,62,63,37,38] }
test = pd.DataFrame.from_dict(dict)

predictions = classifier.predict(input_fn = lambda: input_function(test,False))
predictions