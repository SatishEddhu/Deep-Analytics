import tensorflow as tf
session = tf.InteractiveSession()

d = tf.constant(10,name = "const_d")
e = tf.Variable(d+20, name = "var_e")
f = tf.add(e, tf.constant(1), name = "addition")
session.run(tf.initialize_all_variables())
update = e.assign(e+10)

writer = tf.train.SummaryWriter("output2", session.graph)
writer.close()
session.close()
