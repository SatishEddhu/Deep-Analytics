# Import required packages
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
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

# the arguments 'mode' and 'params' are optional; hence omitted
def model_function(features, targets):
    targets = tf.one_hot(targets,2,1,0) # two perceptrons in output
    
    outputs = layers.fully_connected(inputs=features, 
                                     num_outputs=2,
                                     activation_fn=tf.sigmoid)
    
    outputs_dict = {"labels": outputs}
  
  
    # Calculate loss using mean squared error
    loss = losses.mean_squared_error(outputs, targets)

    # Create training operation
    optimizer = layers.optimize_loss(
                  loss=loss,
                  # step is not an integer but a wrapper around it, just as Java has 'Integer' on top of 'int'
                  global_step=tf.contrib.framework.get_global_step(),
                  learning_rate=0.001,
                  optimizer="SGD")

    # Why return 'loss' separately when it is already a part of optimizer?
    #   evaluate() needs only - outputs_dict,loss [does not need optimizer since it is not learning]
    #   fit() needs only - outputs_dict,loss,optimizer [does not need outputs_dict since it is not predicting]
    #   predict needs only - outputs_dict
    # So, 'loss' sent separately for use by evaluate()
    return outputs_dict, loss, optimizer 

     
# create custom neural network model
classifier = learn.Estimator(model_fn=model_function, model_dir='/home/algo/Algorithmica/tmp')

classifier.fit(x=x_train, y=y_train, steps=2000)
for var in classifier.get_variable_names()    :
    print var, ": ", classifier.get_variable_value(var)

    
#evaluate the model using validation set
results = classifier.evaluate(x=x_validate, y=y_validate, steps=1)
type(results)
for key in sorted(results):
    print "%s:%s" % (key, results[key])
    
# Predict the outcome of test data using model
test = np.array([[100.4,21.5],[200.1,26.1]], dtype=np.float32)
predictions = classifier.predict(test)
predictions