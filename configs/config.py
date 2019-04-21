import os


class Config():
    split = 'vinyals'  # in matching net

    exp_name = 'default'

    num_class_per_episode = 60  # number of classes per episode
    num_sample_per_class = 5  # number of samples per class
    num_query_per_class = 5
    num_episode_per_epoch = 100
    num_episode_per_val_epoch = 25

    sequential_sampler_rather_than_episodic = False
    cuda = False
    model_name = 'protonet_conv'
    image_width = 28
    image_height = 28
    image_channel_size = 1
    hidden_channel_size = 64
    output_channel_size = 64  # the image will be encode as a 64-long vector

    num_epochs = 10000

    @property
    def num_epoch(self):
        return self.num_epochs

    optim_method = 'Adam'
    learning_rate = .001

    learning_rate_decay_period = 20
    weight_decay = 0.0
    patience = 200  # number of epochs to wait before validation improvement (default: 1000)

    fields = 'loss,acc'
    exp_dir = 'experiments'

    mode = 'train'
    max_to_keep = 5

    image_augmentation = True

    @property
    def summary_dir(self):
        return os.path.join("experiments", self.exp_name, "summary/")

    @property
    def checkpoint_dir(self):
        return os.path.join("experiments", self.exp_name, "checkpoint/")

    def print_config(self):


        pass


class OmniglotConfig(Config):
    dataset = 'omniglot'
    num_epoch = 20
    pass


class MiniImageNetConfig(Config):
    dataset = 'miniImageNet'
    exp_name = 'proto_net_2'
    image_width = 84
    image_height = 84
    image_channel_size = 3

    num_epochs = 300
    num_episode_per_epoch = 100
    num_class_per_episode = 5
    num_sample_per_class = 5
    num_query_per_class = 15

    embedding_size = 1600
    learning_rate = .0001
    pass
