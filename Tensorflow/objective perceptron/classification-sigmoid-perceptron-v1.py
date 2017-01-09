# Import required packages
from tensorflow.contrib import learn
from tensorflow.contrib import layers
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sb # for graphics
import os

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("/home/algo/Algorithmica/objective perceptron")
os.getcwd()

# Read the train data
# hascolumns x1(float), x2(float), label(0/1)
sample = pd.read_csv("train.csv")
sample.shape
sample.info()

FEATURES = ['x1','x2']
LABEL = ['label']

sb.swarmplot(x='x1', y='x2', data=sample, hue='label', size=10)

# Preprocess input data
def normalize(x) :    
    return ( (x - np.mean(x)) / (np.max(x) - np.min(x)) )
    
def input_function(dataset, train=False):
    dataset.x1 = normalize(dataset.x1)
    dataset.x2 = normalize(dataset.x2)
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

classifier.weights_
classifier.bias_

# Predict the outcome using model
dict = {'x1':[10.4,21.5,10.5], 'x2':[22.1,26.1,2.7] }
test = pd.DataFrame.from_dict(dict)

predictions = classifier.predict(input_fn = lambda: input_function(test,False))
predictions
