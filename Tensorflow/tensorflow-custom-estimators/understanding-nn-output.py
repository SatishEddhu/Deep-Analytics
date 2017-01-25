# Import required packages
from tensorflow.contrib import layers
from tensorflow.contrib import losses
import tensorflow as tf
import numpy as np
import os

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("/home/algo/Algorithmica/tensorflow-builtin-estimators")

# Learning some basics of neural networks
#########################################

# demonstration of one-hot-encoding in action
o = np.array([0,1,2,1,1,1,1,0])
tmp = tf.one_hot(o,3,1,0) # one-hot(target, #classes, existence, non-existence)
tf.Session().run(tmp)

# sigmoid output. sigmoid(x) = 1/ [1 + e^(-x)]. Range is (0,1)
x2 = np.array([10,100,0,-10,-100],dtype=np.float32)
sg = tf.sigmoid(x2)
tf.Session().run(sg)

# softmax output. softmax(a_j) = e^a_j/ (sigma e^a_k)
# transforms output of 4 perceptrons into probability distribution
y2 = np.array([0.6,0.7,0.9,0.9], dtype=np.float32)
sm = tf.nn.softmax(y2)
tf.Session().run(sm)

features = np.array([[1,2], [3,4]], dtype=np.float32)

# fully connected means, all inputs are connected to every perceptron
# default activation function is 'Linear'
# output is (4,8)
# 4 = 1(1) + 2(1) + 1; 8 = 3(1) + 4(1) + 1
nnout = layers.fully_connected(inputs=features, # [[1,2], [3,4]]
                                  weights_initializer=tf.constant_initializer([1.0]),
                                  biases_initializer=tf.constant_initializer([1.0]),
                                  num_outputs=1,
                                  activation_fn=None)
session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nnout)

# output is sigmoid(4) and sigmoid(8)
nnout2 = layers.fully_connected(inputs=features, # [[1,2], [3,4]]
                                  weights_initializer=tf.constant_initializer([1.0]),
                                  biases_initializer=tf.constant_initializer([1.0]),
                                  num_outputs=1,
                                  activation_fn=tf.sigmoid)
session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nnout2)

# layer of 2 perceptrons. Both of them give the same output of (4,8) and (4,8)
nnout3 = layers.fully_connected(inputs=features, # [[1,2], [3,4]]
                                  weights_initializer=tf.constant_initializer([1.0]),
                                  biases_initializer=tf.constant_initializer([1.0]),
                                  num_outputs=2,
                                  activation_fn=None)
session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nnout3)

# individual weights to each perceptron. Perceptron1 (w11=1,w21=2,bias=1) Perceptron2 (w12=1,w22=2,bias=2)
# output is [6,7] and [12,13]
# Input: 1,2  -->    6 = 1(1) + 2(2) + 1;  7 = 1(1) + 2(2) + 2
# Input: 3,4  -->   12 = 3(1) + 4(2) + 1; 13 = 3(1) + 4(2) + 2
nnout4 = layers.fully_connected(inputs=features, # [[1,2], [3,4]]
                                  # [[w11,w12],[w21,w22]]; wij = weight from ith input to jth perceptron
                                  weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
                                  biases_initializer=tf.constant_initializer([1.0,2.0]),
                                  num_outputs=2,
                                  activation_fn=None)
session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nnout4)

# same as nnout4 but with sigmoid activation. output nnout5 = sigmoid(output of nnout4)
# output = ([ [0.99752742,0.999089], [0.9999938,0.99999774] ]
nnout5 = layers.fully_connected(inputs=features, 
                                  weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
                                  biases_initializer=tf.constant_initializer([1.0,2.0]),
                                  num_outputs=2,
                                  activation_fn=tf.sigmoid)
session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nnout5)

# softmax output. Gives global meaning to both the perceptrons combined
# output is [0.26,0.73], [0.26,0.73].
# [e^6/(e^6+e^7),e^7/(e^6 + e^7) ] = [1/1+e, 1/1+e^-1] = [0.26,0.73]
# [e^12/(e^12+e^13),e^13/(e^12 + e^13) ] = [1/1+e, 1/1+e^-1] = [0.26,0.73]
# Note how the result is very different from that obtained with sigmoid activation function.
nnout6 = layers.fully_connected(inputs=features, 
                                  weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
                                  biases_initializer=tf.constant_initializer([1.0,2.0]),
                                  num_outputs=2,
                                  activation_fn=tf.nn.softmax)
session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nnout6)

outputs = tf.constant([0,1,0,1])
targets = tf.constant([1,1,1,0])
sq_loss1 = losses.mean_squared_error(outputs,targets)
log_loss1 = losses.log_loss(outputs,targets)


outputs2 = tf.constant([[100.0, -100.0, -100.0],
                      [-100.0, 100.0, -100.0],
                      [-100.0, -100.0, 100.0]])
targets2 = tf.constant([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
sq_loss2 = losses.mean_squared_error(outputs2, targets2)

session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(sq_loss1) # 0.75 = [(0-1)^2 + (1-1)^2 + (0-1)^2 + (1-0)^2] / 4
session.run(sq_loss2) # 10067 = (6*100^2 + 3*101^2)/9
session.run(log_loss1) # sigma(-y_i*log(y_i))