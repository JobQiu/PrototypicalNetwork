import numpy as np


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]


class CompressedImageNetDataGenerator:

    def __init__(self, config):
        self.config = config
        self.train_images = np.concatenate([np.load('demo/mini-imagenet-train_{}.npy'.format(i)) for i in range(8)])
        self.test_images = np.concatenate([np.load('demo/mini-imagenet-test_{}.npy'.format(i)) for i in range(2)])

    def next_batch(self):
        config = self.config
        if config.mode == 'train':

            total_num_class = len(self.train_images)
            total_num_sample_per_class = self.train_images.shape[1]

            episode_classes = np.random.permutation(total_num_class)[:config.num_class_per_episode]
            support = np.zeros(shape=[config.num_class_per_episode, config.num_sample_per_class, config.image_height,
                                      config.image_width, config.image_channel_size], dtype=np.float32)

            # if config. image augmentation, use np. flip to get the flip image to feed todo
            # np.flip(A, axis=3)
            query = np.zeros(shape=[config.num_class_per_episode, config.num_query_per_class, config.image_height,
                                    config.image_width, config.image_channel_size], dtype=np.float32)

            for idx, epi_class in enumerate(episode_classes):
                selected = np.random.permutation(total_num_sample_per_class)[
                           :config.num_sample_per_class + config.num_query_per_class]
                support[idx] = self.train_images[epi_class, selected[:config.num_sample_per_class]]
                query[idx] = self.train_images[epi_class, selected[config.num_sample_per_class:]]

            labels = np.tile(np.arange(config.num_class_per_episode)[:, np.newaxis],
                             (1, config.num_query_per_class)).astype(np.uint8)
            yield support, query, labels

        else:
            pass
        pass
