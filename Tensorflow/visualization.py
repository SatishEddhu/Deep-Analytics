import tensorflow as tf

session = tf.InteractiveSession()
a = tf.constant(10, name="constant1")
b = tf.constant(20, name="constant2")
c = tf.add(a,b)

writer = tf.train.SummaryWriter("output", session.graph)
writer.close()
session.close()
