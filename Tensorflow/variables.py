import tensorflow as tf

session = tf.InteractiveSession()

a = tf.Variable(10)
print(type(a))
print(a.get_shape())

b = tf.Variable(tf.zeros(5))
print(type(b))
print(b.get_shape())

c = tf.Variable(tf.zeros((2,3)))
print(type(c))
print(c.get_shape())

#  variables need to be initialised, but constants don't need to
session.run(tf.initialize_all_variables())
print(a.eval())
print(b.eval())
print(c.eval())
# can also evaluate all variables at once as shown below
print(session.run([a,b,c]))


seed = tf.set_random_seed(100)
# floating point random variables
d1 = tf.Variable(tf.random_uniform((10,)))
# we don't use numpy's random, but tf's random to make use
# of tf's functionalities
d2 = tf.Variable(tf.random_uniform((10,),0,2))
# integer random variables
d3 = tf.Variable(tf.random_uniform(
            shape=(10,),minval=1,maxval=100,dtype=tf.int32) )

session.run(tf.initialize_all_variables())
print(d1.eval())
print(d2.eval())
print(d3.eval())

d = tf.constant(10)
e = tf.Variable(d+20)
f = tf.add(e, tf.constant(1))
session.run(tf.initialize_all_variables())
print(session.run([d,e,f]))
update = e.assign(e+10)
update.eval()
print(session.run([d,e,f]))

print(session.run([d,e,f,update]))
