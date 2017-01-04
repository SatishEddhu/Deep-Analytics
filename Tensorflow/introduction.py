import numpy
import tensorflow as tf

# without any explicit execution of operations
c1 = tf.constant(100) # 0-dimensional array
type(c1)
c1.get_shape()
print(c1) # getting only object details, not values
c1

# execute explicitly with session
session1 = tf.Session()
print(session1.run(c1))
session1.close()

# execute implicitly with session; more convenient to program
session2 = tf.InteractiveSession()
print(c1.eval()) # eval is shortcut for session.run() method
session2.close()

