import tensorflow as tf
import numpy as np

session = tf.InteractiveSession()

# rank 0 tensor
ct1 = tf.constant(10)
ct1.get_shape()
type(ct1)
ct1.eval()

# rank 1 tensor
ct2 = tf.constant([10,20,30])
ct2.get_shape()
ct2.eval()

ct3 = tf.constant(np.array([20,30]))
ct3.get_shape() # although says Dimension(3), it is a 1-D tensor only
ct3.eval()

# rank 2 tensor
ct4 = tf.constant(np.array([[20,30],[50,60]]))
ct4.get_shape() # Dimension(2), Dimension(2)
ct4.eval()

# giving name to node (not name to constant)
ct5 = tf.constant(100, name="constant")
ct5.get_shape()
ct5.eval()

ct6 = tf.constant(10,shape=[3])
print(ct6.eval())

ct7 = tf.constant(-1,shape=[2,3])
print(ct7.eval())

z1 = tf.zeros(5, tf.int32) # default type is float; avoid by specifying type explicitly
type(z1)
z1.get_shape()
z1.eval()

z2 = tf.zeros((2,2))
type(z2)
z2.get_shape()
z2.eval()

session.close()
