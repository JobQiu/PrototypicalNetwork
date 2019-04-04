from base.base_model import BaseModel
import tensorflow as tf
from utils.tf_utils import euclidean_distance


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


class PrototypicalNetwork(BaseModel):

    def __init__(self, config):
        super(PrototypicalNetwork, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        config = self.config
        x = tf.placeholder(dtype=tf.float32,
                           shape=[None, None, config.image_height, config.image_width, config.image_channel_size])
        self.x = x
        q = tf.placeholder(dtype=tf.float32,
                           shape=[None, None, config.image_height, config.image_width, config.image_channel_size])
        self.q = q
        x_shape = tf.shape(x)
        q_shape = tf.shape(q)
        num_classes, num_support = x_shape[0], x_shape[1]
        num_queries = q_shape[1]
        y = tf.placeholder(tf.int64, [None, None])
        self.y = y
        y_one_hot = tf.one_hot(y, depth=num_classes)
        emb_in = encoder(tf.reshape(x, [num_classes * num_support, config.image_height, config.image_width,
                                        config.image_channel_size]), config.hidden_channel_size,
                         config.output_channel_size)
        emb_dim = tf.shape(emb_in)[-1]
        self.emb_x = emb_in
        emb_x = tf.reduce_mean(tf.reshape(emb_in, [num_classes, num_support, emb_dim]), axis=1)
        emb_q = encoder(tf.reshape(q, [num_classes * num_queries, config.image_height, config.image_width,
                                       config.image_channel_size]),
                        config.hidden_channel_size,
                        config.output_channel_size,
                        reuse=True)
        self.emb_q = emb_q
        self.prototype = emb_x
        dists = euclidean_distance(emb_q, emb_x)
        log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])
        # cross entropy loss
        ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
        acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))
        self.loss = ce_loss
        self.acc = acc
        self.train_op = tf.train.AdamOptimizer().minimize(ce_loss)


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
