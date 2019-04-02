class Config():
    split = 'vinyals'  # in matching net
    num_class_per_episode = 60  # number of classes per episode
    num_sample_per_class = 5  # number of samples per class
    num_query_per_class = 5
    num_episode_per_epoch = 100

    sequential_sampler_rather_than_episodic = False
    cuda = False
    model_name = 'protonet_conv'
    image_width = 28
    image_height = 28
    image_channel_size = 1
    hidden_channel_size = 64
    output_channel_size = 64  # the image will be encode as a 64-long vector

    num_epoch = 10000
    optim_method = 'Adam'
    learning_rate = .001

    learning_rate_decay_period = 20
    weight_decay = 0.0
    patience = 200  # number of epochs to wait before validation improvement (default: 1000)

    fields = 'loss,acc'
    exp_dir = 'experiments'

    pass


class OmniglotConfig(Config):
    dataset = 'omniglot'

    num_epoch = 20

    pass


class MiniImageNetConfig(Config):
    dataset = 'miniImageNet'
    image_width = 84
    image_height = 84
    image_channel_size = 3
    pass
