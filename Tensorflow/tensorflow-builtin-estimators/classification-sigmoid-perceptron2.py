# Import required packages
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from sklearn import model_selection
import tensorflow as tf
import numpy as np
import os

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("/home/algo/Algorithmica/tensorflow-builtin-estimators")

#load the dataset
sample = learn.datasets.base.load_csv_with_header(
      filename="train2.csv",
      target_dtype=np.int,
      features_dtype=np.float32, target_column=-1)

x = sample.data
y = sample.target

# Divide the input data into train and validation
x_train,x_validate,y_train,y_validate = model_selection.train_test_split(x, y, test_size=0.2, random_state=100)
type(x_train)

len(x_train)
len(y_train)
len(x_validate)
len(y_validate)

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

#evaluate the model using validation set
results = classifier.evaluate(x=x_validate, y=y_validate, steps=1)
type(results)
for key in sorted(results):
    print "%s:%s" % (key, results[key])
    
# Predict the outcome of test data using model
test = np.array([[100.4,21.5,10.5,22.4],[200.1,26.1,2.7,26.7]])
predictions = classifier.predict(test)
predictions
