import tensorflow as tf

from configs.config import MiniImageNetConfig
from data_loader.data_generator import CompressedImageNetDataGenerator
from models.example_model import PrototypicalNetwork
from trainers.example_trainer import ProtoNetTrainer
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import send_msg

if __name__ == "__main__":
    test = True

    config = MiniImageNetConfig()
    config.exp_name = "proto_net_2"
    config.learning_rate = .0005
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create your data generator
    data = CompressedImageNetDataGenerator(config)
    model = PrototypicalNetwork(config)

    sess = tf.Session()
    logger = Logger(sess, config)
    trainer = ProtoNetTrainer(sess, model, data, config, logger)
    model.load(sess)
    if test:
        test_loss, test_acc = trainer.test()
        print("test result")
        print("test_loss = {}".format(test_loss))
        print("test_acc = {}".format(test_acc))

    else:

        trainer.train()

        send_msg("Done")
