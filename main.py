import tensorflow as tf
import os

from configs.config import MiniImageNetConfig
from data_loader.data_generator import DataGenerator, CompressedImageNetDataGenerator
from models.example_model import ExampleModel, PrototypicalNetwork, PrototypicalNetwork2
from trainers.example_trainer import ExampleTrainer, ProtoNetTrainer, ProtoNetTrainer2
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args, send_msg


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        if os.path.isfile("configs/example.json"):
            config = process_config("configs/example.json")
        else:
            config = process_config("../configs/example.json")

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)

    # create an instance of the model you want
    model = ExampleModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


def run_proto_net():
    config = MiniImageNetConfig()
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create your data generator
    data = CompressedImageNetDataGenerator(config)
    model = PrototypicalNetwork2(config)

    sess = tf.Session()
    logger = Logger(sess, config)
    trainer = ProtoNetTrainer2(sess, model, data, config, logger)
    model.load(sess)
    trainer.train()

    pass


def generate_image_embedding():
    config = MiniImageNetConfig()
    create_dirs([config.summary_dir, config.checkpoint_dir])

    data = CompressedImageNetDataGenerator(config)
    model = PrototypicalNetwork(config)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    model.load(sess)

    train_inputs, train_query, train_labels = next(data.next_batch())
    x_embedding, q_embedding, prototype = sess.run(fetches=[model.embedded_x, model.embedded_q, model.prototype],
                                                   feed_dict={model.inputs: train_inputs,
                                                              model.query: train_query,
                                                              model.labels: train_labels})

    print("")

    pass


if __name__ == '__main__':

    experiment = 'protoNet'

    if experiment == 'protoNet':
        run_proto_net()
    elif experiment == 'protoNet_embedding':
        generate_image_embedding()
    else:
        main()

    send_msg("train done")
