import tensorflow as tf


def euclidean_distance(a, b):
    """
    firstly expand a's shape to be
    (N, 1, D) and then tile it to be
    (N, M, D) by copy a M times
    and then expand b's shape to be
    (1, M, D) and then tile it to be
    (N, M, D) by copying b N times

    then do the element wise minus and square and reduce mean,
    we will have a N by M matrix, to be the
    distance matrix

    :param a: shape is N row and D for each vector
    :param b: shape is M row and D for each vecto
    :return:
    return the e distance between every row and every row in b
    """

    N, D = a.get_shape().as_list()
    M = b.get_shape().as_list()[0]

    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))

    return tf.reduce_mean(tf.square(a - b), axis=2)

# %%
