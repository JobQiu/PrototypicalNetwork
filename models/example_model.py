from base.base_model import BaseModel
import tensorflow as tf
from utils.tf_utils import euclidean_distance


class PrototypicalNetwork(BaseModel):

    def __init__(self, config):
        super(PrototypicalNetwork, self).__init__(config)
        self.config = config
        self.build_model()
        pass

    def build_model(self, backbone='base', num_core=1):
        """

        :param backbone:
        :return:
        """

        config = self.config
        height, width, num_channel = config.image_height, config.image_width, config.image_channel_size
        num_class = config.num_class_per_episode
        num_sample_per_class = config.num_sample_per_class
        num_query_per_class = config.num_query_per_class

        self.inputs = tf.placeholder(tf.float32, [num_class, num_sample_per_class, height, width, num_channel])
        self.query = tf.placeholder(tf.float32, [num_class, num_query_per_class, height, width, num_channel])
        self.labels = tf.placeholder(tf.int64, [num_class, num_query_per_class])
        self.labels_one_hot = tf.one_hot(self.labels, depth=num_class)

        if backbone == 'base':
            embedded_x = self._build_base_encoder(
                tf.reshape(self.inputs, shape=[num_class * num_sample_per_class, height, width, num_channel]),
                config.hidden_channel_size, config.output_channel_size)
            self.embedded_x = embedded_x
            embedded_q = self._build_base_encoder(
                tf.reshape(self.query, shape=[num_class * num_query_per_class, height, width, num_channel]),
                config.hidden_channel_size, config.output_channel_size,
                reuse=True)
            self.embedded_q = embedded_q
            embedding_size = tf.shape(embedded_x)[-1]
        else:
            raise NotImplementedError

        if num_core == 1:
            prototype = tf.reshape(tensor=embedded_x, shape=[num_class, num_sample_per_class, embedding_size])
            prototype = tf.reduce_mean(prototype, axis=1)  # the average of all embedding of this class
            self.prototype = prototype
        else:
            raise NotImplementedError

        distance_matrix = euclidean_distance(prototype, embedded_q)
        log_p_y = tf.reshape(tf.nn.log_softmax(-distance_matrix), [num_class, num_query_per_class, -1])
        self.loss = -tf.reduce_mean(
            tf.reshape(tf.reduce_sum(tf.multiply(self.labels_one_hot, log_p_y), axis=-1), [-1]))
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), self.labels)))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step_tensor)

    def _build_base_encoder(self, inputs, num_hidden_channel, num_output_channel, reuse=False):
        """ encode the input images
        :param inputs: [number_class, num_smaple_per_class, width, height, num_channels]
        :param num_hidden_channel:
        :param num_output_channel:
        :return: a flat vector(1-d embedding) per image
        """

        def conv_block(inputs,
                       out_channels,
                       layer=1,
                       filter_size=3):
            with tf.variable_scope(name_or_scope="conv%d" % layer):
                conv = tf.layers.conv2d(inputs, out_channels, kernel_size=filter_size, padding='SAME')
                conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=.99, scale=True, center=True)
                conv = tf.nn.relu(conv)
                conv = tf.contrib.layers.max_pool2d(conv, 2)
                return conv

        with tf.variable_scope(name_or_scope="encoder", reuse=reuse):
            net = conv_block(inputs, num_hidden_channel, layer=1)
            net = conv_block(net, num_hidden_channel, layer=2)
            net = conv_block(net, num_hidden_channel, layer=3)
            net = conv_block(net, num_hidden_channel, layer=4)
            net = conv_block(net, num_output_channel, layer=5)
            net = tf.contrib.layers.flatten(net)
            return net

    def _build_encoder_with_backbone(self):
        pass


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
