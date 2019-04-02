from base.base_model import BaseModel
import tensorflow as tf


class PrototypicalNetwork(BaseModel):

    def __init__(self, config):
        self.config = config

        pass

    def build_model(self, backbone='base'):
        if backbone == 'base':
            pass
        else:
            pass
        pass

    def _build_base_model(self, inputs, hidden_size, z_dim, reuse=False):
        """

        :param input:
        :param hidden_size:
        :param z_dim:
        :param reuse:
        :return:
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

        with tf.variable_scope(name_or_scope="", reuse=reuse):
            net = conv_block(inputs, hidden_size, layer=1)
            net = conv_block(net, hidden_size, layer=2)
            net = conv_block(net, hidden_size, layer=3)
            net = conv_block(net, hidden_size, layer=4)
            net = conv_block(net, z_dim, layer=5)
            return net

    def _build_model_with_backbone(self):
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
