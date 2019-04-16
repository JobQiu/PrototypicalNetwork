from base.base_model import BaseModel
import tensorflow as tf
from utils.tf_utils import euclidean_distance, euclidean_distance_with_weight


def conv_block(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        return conv


def encoder(x, h_dim, z_dim, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = conv_block(x, h_dim, name='conv_1')
        net = conv_block(net, h_dim, name='conv_2')
        net = conv_block(net, h_dim, name='conv_3')
        net = conv_block(net, z_dim, name='conv_4')
        net = tf.contrib.layers.flatten(net)
        return net


def embedding2weights(x, num_class=20, num_support=5, embedding_size=1600):
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
        out = tf.scalar_mul(1600, out)
        # out = tf.multiply(out, embedding_size)
        return out


class PrototypicalNetwork(BaseModel):

    def __init__(self, config):
        super(PrototypicalNetwork, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        config = self.config

        num_class = config.num_class_per_episode
        num_support = config.num_sample_per_class
        num_query = config.num_query_per_class

        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[num_class, num_support, config.image_height, config.image_width,
                                       config.image_channel_size],
                                name='support_set')

        self.q = tf.placeholder(dtype=tf.float32,
                                shape=[num_class, num_query, config.image_height, config.image_width,
                                       config.image_channel_size],
                                name='query')

        self.y = tf.placeholder(tf.int64, [None, None], name='label_of_query')
        y_one_hot = tf.one_hot(self.y, depth=num_class)
        self.emb_x = encoder(tf.reshape(self.x, [num_class * num_support, config.image_height, config.image_width,
                                                 config.image_channel_size]), config.hidden_channel_size,
                             config.output_channel_size)

        emb_dim = tf.shape(self.emb_x)[-1]

        weights = embedding2weights(self.emb_x, num_class, num_support,
                                    embedding_size=emb_dim)  # embedding_size=config.embedding_size)

        self.prototype = tf.reduce_mean(tf.reshape(self.emb_x, [num_class, num_support, emb_dim]), axis=1,
                                        name='prototype')
        self.emb_q = encoder(tf.reshape(self.q, [num_class * num_query, config.image_height, config.image_width,
                                                 config.image_channel_size]),
                             config.hidden_channel_size,
                             config.output_channel_size,
                             reuse=True)

        # dists = euclidean_distance(self.emb_q, self.prototype)
        dists = euclidean_distance_with_weight(self.emb_q, self.prototype, weights)
        log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_class, num_query, -1])
        # cross entropy loss
        self.loss = -tf.reduce_mean(
            tf.reshape(
                tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1),
                [-1]
            ),
            name='loss'
        )

        self.acc = tf.reduce_mean(tf.cast(x=tf.equal(tf.argmax(log_p_y, axis=-1), self.y),
                                          dtype=tf.float32
                                          ), name='accuracy'
                                  )
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step_tensor)


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # network architecture
        d1 = tf.layers.dense(self.x, 512, activation=tf.nn.relu, name="dense1")
        d2 = tf.layers.dense(d1, 10, name="dense2")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


class PrototypicalNetwork_v2(BaseModel):

    def __init__(self, config):
        super(PrototypicalNetwork_v2, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        config = self.config

        self.support_set = tf.placeholder(dtype=tf.float32,
                                          shape=[None, None, config.image_height, config.image_width,
                                                 config.image_channel_size],
                                          name='support_set')
        self.queqy = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None, config.image_height, config.image_width,
                                           config.image_channel_size],
                                    name='query')
        self.query_label = tf.placeholder(dtype=tf.int32, shape=[None, None], name='label')

        query_label_one_hot = tf.one_hot(self.query_label, depth=config.num_class_per_episode)

        pass
