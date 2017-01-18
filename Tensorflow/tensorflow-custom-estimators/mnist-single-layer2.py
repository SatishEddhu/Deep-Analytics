from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

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

type(x_train)
type(x_validation)
type(x_test)
x_train.shape
x_train.shape[0]
x_validation.shape
x_test.shape
y_train.shape

def explore_data(features, targets):
    # shape is tuple; shape[0] contains the number of rows (input size)
    randidx = np.random.randint(features.shape[0], size=5) # pick 5 random rows
    for i in randidx:
        curr_img = np.reshape(features[i, :], (28,28)) # reshape 784 pixels to 28 x 28 format for display
        curr_label = np.argmax(targets[i,:]) # get the place where '1' is there in the one-hot-coded columns; one-hot -> original class
        plt.matshow(curr_img,cmap=plt.get_cmap('gray'))
        print("" + str(i) + "th training data, " + "label is " + str(curr_label))

explore_data(x_train,y_train)


def model_function(features, targets, mode):
    # don't need one-hot encoding since target is already in one-hot format
    
    # sigmoid also will work although the interpretability is difficult;
    # The output with the max. value corresponds to the 'class' - whether sigmoid or softmax
    outputs = layers.fully_connected(inputs=features, 
                                     num_outputs=10, # 10 perceptrons for 10 numbers (0 to 9)
                                     activation_fn=None) # Use "None" as activation function specified in "softmax_cross_entropy" loss
    # layer gives direct/plain outputs - linear activation. To compute losses, we use softmax on top of plain outputs
    
    
    # Calculate loss using cross-entropy error; also use the 'softmax' activation function
    # softmax and cross-entropy combined together to handle log(0) and other border-case issues
    loss = losses.softmax_cross_entropy (outputs, targets)
    
    optimizer = layers.optimize_loss(
                  loss=loss,
                  # step is not an integer but a wrapper around it, just as Java has 'Integer' on top of 'int'
                  global_step=tf.contrib.framework.get_global_step(),
                  learning_rate=0.001,
                  optimizer="SGD")

    # Class of output (i.e., predicted number) corresponds to the perceptron returning the highest fractional value
    # Returning both fractional values and corresponding labels    
    return {'probs':tf.nn.softmax(outputs), 'labels':tf.argmax(tf.nn.softmax(outputs), 1)}, loss, optimizer 
    # Applying softmax on top of plain outputs from layer (linear activation function since activation_fn=None) to give results
    
    
classifier = learn.Estimator(model_fn=model_function, model_dir='/home/algo/Algorithmica/tmp')

# 784 x 10 weights involved and adjusted each time across 55000 inputs; hence takes a long time
# Number of inputs for each step = 55000/1000 = 55 (stochastic mini-batch)
# We must cover all inputs in an epoch. 1000 steps in an epoch. So,55 inputs in each step
classifier.fit(x=x_train, y=y_train, steps=1000)
for var in classifier.get_variable_names()    :
    print var, ": ", classifier.get_variable_value(var)

#evaluate the model using validation set
results = classifier.evaluate(x=x_validation, y=y_validation, steps=1)
type(results)
for key in sorted(results):
    print "%s:%s" % (key, results[key])
    
# Predict the outcome of test data using model
predictions = classifier.predict(x_test, as_iterable=True)
for i, p in enumerate(predictions):
   print("Prediction %s: %s, probs = %s" % (i+1, p["labels"], p["probs"]))
