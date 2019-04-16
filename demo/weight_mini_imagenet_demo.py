# coding: utf-8

# In[1]:


from __future__ import print_function

import numpy as np
import tensorflow as tf


# In[2]:


def conv_block(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        return conv


# In[3]:


def encoder(x, h_dim, z_dim, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = conv_block(x, h_dim, name='conv_1')
        net = conv_block(net, h_dim, name='conv_2')
        net = conv_block(net, h_dim, name='conv_3')
        net = conv_block(net, z_dim, name='conv_4')
        net = tf.contrib.layers.flatten(net)
        return net


# In[4]:


def euclidean_distance(a, b):
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


def embedding2weights(x, num_class=20, num_support=5):
    if len(x.get_shape()) == 2:
        x = tf.reshape(x, [num_class, num_support, -1])

    with tf.variable_scope(name_or_scope="get_weight", reuse=tf.AUTO_REUSE):
        x_max = tf.expand_dims(tf.reduce_max(x, 1), 1)
        x_min = tf.expand_dims(tf.reduce_min(x, 1), 1)
        x_sum = tf.expand_dims(tf.reduce_sum(x, 1), 1)
        x_prod = tf.expand_dims(tf.reduce_prod(x, 1), 1)
        x_mean, x_variance = tf.nn.moments(x, [1])
        x_mean = tf.expand_dims(x_mean, 1)
        x_variance = tf.expand_dims(x_variance, 1)

        para_list = [x_max, x_min, x_mean, x_prod, x_sum, x_variance]

        x_all = tf.concat(para_list, 1)
        x_all = tf.transpose(x_all, perm=[0, 2, 1])

        weight = tf.get_variable(shape=(len(para_list), 1), name='weight', dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        _W_t = tf.tile(tf.expand_dims(weight, axis=0), [num_class, 1, 1])

        out = tf.matmul(x_all, _W_t)
        out = tf.squeeze(out, axis=2)
        out = tf.nn.softmax(out, axis=1)
        print(out.get_shape())
        out = tf.scalar_mul(1600, out)
        return out


# In[5]:


n_epochs = 100
n_episodes = 100
n_way = 20
n_shot = 5
n_query = 15
n_examples = 350
im_width, im_height, channels = 84, 84, 3
h_dim = 64
z_dim = 64

# In[6]:


# Load Train Dataset
train_dataset = []
for i in range(8):
    train_dataset.append(np.load('demo/mini-imagenet-train_{}.npy'.format(i)))
train_dataset = np.concatenate(train_dataset)

n_classes = train_dataset.shape[0]
print(train_dataset.shape)

# In[7]:


x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
x_shape = tf.shape(x)
q_shape = tf.shape(q)
num_classes, num_support = x_shape[0], x_shape[1]
num_queries = q_shape[1]
y = tf.placeholder(tf.int64, [None, None])
y_one_hot = tf.one_hot(y, depth=num_classes)
emb_in = encoder(tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), h_dim, z_dim)
weights = embedding2weights(emb_in, num_classes, num_support)
emb_dim = tf.shape(emb_in)[-1]
emb_x = tf.reduce_mean(tf.reshape(emb_in, [num_classes, num_support, emb_dim]), axis=1)
emb_q = encoder(tf.reshape(q, [num_classes * num_queries, im_height, im_width, channels]), h_dim, z_dim, reuse=True)
# dists = euclidean_distance(emb_q, emb_x)
dists = euclidean_distance_with_weight(emb_q, emb_x, weights)

log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])
ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))

# In[8]:


train_op = tf.train.AdamOptimizer().minimize(ce_loss)

# In[9]:


sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

# In[10]:


for ep in range(n_epochs):
    for epi in range(n_episodes):
        epi_classes = np.random.permutation(n_classes)[:n_way]
        support = np.zeros([n_way, n_shot, im_height, im_width, channels], dtype=np.float32)
        query = np.zeros([n_way, n_query, im_height, im_width, channels], dtype=np.float32)
        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_examples)[:n_shot + n_query]
            support[i] = train_dataset[epi_cls, selected[:n_shot]]
            query[i] = train_dataset[epi_cls, selected[n_shot:]]
        # support = np.expand_dims(support, axis=-1)
        # query = np.expand_dims(query, axis=-1)
        labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
        _, ls, ac = sess.run([train_op, ce_loss, acc], feed_dict={x: support, q: query, y: labels})
        if (epi + 1) % 50 == 0:
            print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep + 1, n_epochs, epi + 1,
                                                                                     n_episodes, ls, ac))

# In[11]:


# Load Test Dataset
test_dataset = np.concatenate([np.load('demo/mini-imagenet-test_0.npy'), np.load('demo/mini_imagenet-test_1.npy')])
n_test_classes = test_dataset.shape[0]
print(test_dataset.shape)

# In[12]:


n_test_episodes = 600
n_test_way = 5
n_test_shot = 5
n_test_query = 15

# In[13]:


print('Testing...')
avg_acc = 0.
for epi in range(n_test_episodes):
    epi_classes = np.random.permutation(n_test_classes)[:n_test_way]
    support = np.zeros([n_test_way, n_test_shot, im_height, im_width, channels], dtype=np.float32)
    query = np.zeros([n_test_way, n_test_query, im_height, im_width, channels], dtype=np.float32)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(n_examples)[:n_test_shot + n_test_query]
        support[i] = test_dataset[epi_cls, selected[:n_test_shot]]
        query[i] = test_dataset[epi_cls, selected[n_test_shot:]]
    # support = np.expand_dims(support, axis=-1)
    # query = np.expand_dims(query, axis=-1)
    labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
    ls, ac = sess.run([ce_loss, acc], feed_dict={x: support, q: query, y: labels})
    avg_acc += ac
    if (epi + 1) % 50 == 0:
        print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi + 1, n_test_episodes, ls, ac))
avg_acc /= n_test_episodes
print('Average Test Accuracy: {:.5f}'.format(avg_acc))
