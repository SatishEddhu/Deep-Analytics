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
        features_dtype=np.float32, target_column=-1)

type(sample)
sample.data
sample.data.shape
type(sample.data)
sample.target
sample.target.shape
type(sample.target)

#feature_columns argument expects list of tensorflow feature types
feature_cols = [layers.real_valued_column("", dimension=2)]

classifier = learn.LinearClassifier(feature_columns=feature_cols,
                                            n_classes=2,
                                            model_dir="/home/algo/Algorithmica/tmp")

classifier.fit(x=sample.data, y=sample.target, steps=1000)

#access the learned model parameters
classifier.weights_
classifier.bias_

for var in classifier.get_variable_names():
    print classifier.get_variable_value(var)
    
# w1*x + w2*y + b = 0.
p1 = [0,-classifier.bias_[0]/classifier.weights_[1]] # (0, -b/w2)
p2 = [-classifier.bias_[0]/classifier.weights_[0],0] # (-b/w1, 0)

df = pd.DataFrame(data=np.c_[sample.data, sample.target.astype(int)], columns=['x1','x2','label'])
sb.swarmplot(x='x1', y='x2', data=df, hue='label', size=10)
plt.plot(p1, p2, 'b-', linewidth = 2)

# predict the outcome using model                                  
test = np.array([[100.4,21.5,10.5,22.4],[200.1,26.1,2.7,26.7]])
predictions = classifier.predict(test)
predictions
