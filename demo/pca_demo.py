import tensorflow as tf

inputs = tf.placeholder(tf.float32, shape=())  # type, shape, name
s, u, v = tf.svd()
