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
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)


def euclidean_distance_with_weight(a, b, weight):
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

    :param a: shape is N row and D for each vector, for query
    :param b: shape is M row and D for each vector, for prototype
    :return:
    return the e distance between every row and every row in b
    """
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    weight = tf.tile(tf.expand_dims(weight, axis=0), (N, 1, 1))

    square = tf.square(a - b)
    weight_square = tf.multiply(weight, square)
    return tf.reduce_mean(weight_square, axis=2)
