import tensorflow as tf
session = tf.InteractiveSession()

# equivalent to scanf() in C language
# 0-D tensor
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = tf.mul(x,y)
# actual values supplied
print(session.run(z, feed_dict={x:2,y:3}))

# 1-D tensor
a = tf.placeholder(tf.int32, (2,))
b = tf.placeholder(tf.int32, (2,))
c = tf.add(a,b)
print(session.run(c, feed_dict={a:[10,20],b:[30,40]}))

# 2-D tensor
a = tf.placeholder(tf.int32, (2,2))
b = tf.placeholder(tf.int32, (2,2))
c = tf.add(a,b)
print(session.run(c, feed_dict={a:[[10,20],[30,40]],b:[[30,40],[50,60]]}))
