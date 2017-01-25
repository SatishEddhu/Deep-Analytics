from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.INFO)

# read digit images of 28 x 28 = 784 pixels size
# target is image value in [0,9] range; one-hot encoded to 10 columns
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_validation = mnist.validation.images
y_validation = mnist.validation.labels

x_test = mnist.test.images
y_test = mnist.test.labels

def model_function(features, targets, mode):
    # don't need one-hot encoding since target is already in one-hot format
    
    # sigmoid also will work although the interpretability is difficult;
    # The output with the max. value corresponds to the 'class' - whether sigmoid or softmax
    outputs = layers.fully_connected(inputs=features, 
                                     num_outputs=10, # 10 perceptrons for 10 numbers (0 to 9)
                                     activation_fn=None) # Use "None" as activation function specified in "sigmoid_cross_entropy" loss
    # layer gives direct/plain outputs - linear activation. To compute losses, we use softmax on top of plain outputs
    
    
    # Calculate loss using cross-entropy error; also use the 'sigmoid' activation function
    # sigmoid and cross-entropy combined together to handle log(0) and other border-case issues
    loss = losses.sigmoid_cross_entropy (outputs, targets)
    
    optimizer = layers.optimize_loss(
                  loss=loss,
                  # step is not an integer but a wrapper around it, just as Java has 'Integer' on top of 'int'
                  global_step=tf.contrib.framework.get_global_step(),
                  learning_rate=0.5,
                  optimizer="SGD")

    # Class of output (i.e., predicted number) corresponds to the perceptron returning the highest fractional value
    # Returning both fractional values and corresponding labels    
    probs = tf.sigmoid(outputs)
    return {'probs':probs, 'labels':tf.argmax(probs, 1)}, loss, optimizer 
    # Applying sigmoid on top of plain outputs from layer (linear activation function since activation_fn=None) to give results
    
    
classifier = learn.Estimator(model_fn=model_function, model_dir='/home/algo/Algorithmica/tmp')

classifier.fit(x=x_train, y=y_train, steps=5000, batch_size=100)
for var in classifier.get_variable_names()    :
    print var, ": ", classifier.get_variable_value(var).shape, " - ", classifier.get_variable_value(var)
    #print var, ": ", classifier.get_variable_value(var)

#evaluate the model using validation set
results = classifier.evaluate(x=x_validation, y=y_validation, steps=1)
type(results)
for key in sorted(results):
    print "%s:%s" % (key, results[key])
    
# Predict the outcome of test data using model
predictions = classifier.predict(x_test, as_iterable=True)
for i, p in enumerate(predictions):
   print("Prediction %s: %s, probs = %s" % (i+1, p["labels"], p["probs"]))

# 91%  accuracy with just a single layer
predictions = classifier.predict(x_test)
metrics.accuracy_score(np.argmax(y_test, 1), predictions['labels'])

# check for overfitting; 90.3% accuracy on 'train' and 90.8% on 'validation' implies 'no overfitting'
# Overfitting is when there is high accuracy on 'train' and low accuracy on 'validation'
predictions = classifier.predict(x_train)
metrics.accuracy_score(np.argmax(y_train, 1), predictions['labels'])
predictions = classifier.predict(x_validation)
metrics.accuracy_score(np.argmax(y_validation, 1), predictions['labels'])
