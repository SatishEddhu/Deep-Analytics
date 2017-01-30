from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import metrics
import numpy as np

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


# Hidden layers generally use sigmoid perceptrons
# Output layer uses softmax for overall interpretability of all the 10 outputs
def model_function(features, targets, mode):
    
    # input layer
    # Reshape features to 4-D tensor (55000x28x28x1)
    # MNIST images are 28x28 pixels
    # batch size corresponds to number of images: -1 represents ' compute the # images automatically (55000)'
    # +1 represents the # channels. Here #channels =1 since grey image. For color image, #channels=3
    input_layer = tf.reshape(features, [-1,28,28,1])
    
    
    # Computes 32 features using a 5x5 filter
    # Padding is added to preserve width
    # Input Tensor Shape: [batch_size,28,28,1]
    # Output Tensor Shape: [batch_size,28,28,32]
    conv1 = layers.conv2d(
                inputs=input_layer,
                num_outputs=32,
                kernel_size=[5,5],
                stride=1,
                padding="SAME", # do so much padding such that the feature map is same size as input
                activation_fn=tf.nn.relu)
    
    # Pooling layer 1
    # Pooling layer ith a 2x2 filter and stride 2
    # Input shape: [batch_size,28,28,32]
    # Output shape: [batch_size,14,14,32]
    pool1 = layers.max_pool2d(inputs=conv1,kernel_size=[2,2], stride=2)
    
    # Convolution layer 2
    # Input: 14 x 14 x 32 (32 channels here)
    # Output: 14 x 14 x 64  (32 features/patches fed to each perceptron; discovering 64 features)
    conv2 = layers.conv2d(
                inputs=pool1,
                num_outputs=64,
                kernel_size=[5,5],
                stride=1,
                padding="SAME", # do so much padding such that the feature map is same size as input
                activation_fn=tf.nn.relu)
    
    # Pooling layer 2
    # Input: 14 x14 x 64
    # Output: 7 x 7 x 64
    pool2 = layers.max_pool2d(inputs=conv2,kernel_size=[2,2], stride=2)
    
     
    # Flatten the pool2 to feed to the 1st layer of fully connected layers
    # Input size: [batch_size,7,7,64]
    # Output size: [batch_size, 7x7x64]
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])
         
     
    # Connected layers with 100, 20 neurons
    # Input shape: [batch_size, 7x7x64]
    # Output shape: [batch_size, 10]
    fclayers = layers.stack(pool2_flat, layers.fully_connected, [100,20], 
                             activation_fn=tf.nn.relu, weights_regularizer=layers.l1_l2_regularizer(1.0,2.0),
                             weights_initializer=layers.xavier_initializer(uniform=True,seed=100))
    
    
    outputs = layers.fully_connected(inputs=fclayers, 
                                     num_outputs=10, # 10 perceptrons in output layer for 10 numbers (0 to 9)
                                     activation_fn=None) # Use "None" as activation function specified in "softmax_cross_entropy" loss
    
    
    # Calculate loss using cross-entropy error; also use the 'softmax' activation function
    loss = losses.softmax_cross_entropy (outputs, targets)
    
    optimizer = layers.optimize_loss(
                  loss=loss,                  
                  global_step=tf.contrib.framework.get_global_step(),
                  learning_rate=0.1,
                  optimizer="SGD")

    # Class of output (i.e., predicted number) corresponds to the perceptron returning the highest fractional value
    # Returning both fractional values and corresponding labels    
    probs = tf.nn.softmax(outputs)
    return {'probs':probs, 'labels':tf.argmax(probs, 1)}, loss, optimizer 
    # Applying softmax on top of plain outputs from layer (linear activation function since activation_fn=None) to give results
    
    
classifier = learn.Estimator(model_fn=model_function, model_dir='/home/algo/Algorithmica/tmp')

classifier.fit(x=x_train, y=y_train, steps=2000, batch_size=100)

for var in classifier.get_variable_names()    :
    print var, ": ", classifier.get_variable_value(var).shape, " - ", classifier.get_variable_value(var)

#evaluate the model using validation set
results = classifier.evaluate(x=x_validation, y=y_validation, steps=1)
type(results)
for key in sorted(results):
    print "%s:%s" % (key, results[key])
    
# Predict the outcome of test data using model
predictions = classifier.predict(x_test, as_iterable=True)
for i, p in enumerate(predictions):
   print("Prediction %s: %s, probs = %s" % (i+1, p["labels"], p["probs"]))

# Compute the accuracy metrics
# call with as_iterable=False to get all predictions together
predictions = classifier.predict(x_test)
metrics.accuracy_score(np.argmax(y_test, 1), predictions['labels'])

# checking how well this fit the train data
predictions = classifier.predict(x_train)
metrics.accuracy_score(np.argmax(y_train, 1), predictions['labels'])
