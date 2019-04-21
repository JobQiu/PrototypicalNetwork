import tensorflow as tf
import os

from configs.config import MiniImageNetConfig
from data_loader.data_generator import DataGenerator, CompressedImageNetDataGenerator
from models.example_model import ExampleModel, PrototypicalNetwork
from trainers.example_trainer import ExampleTrainer, ProtoNetTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args, send_msg

config = MiniImageNetConfig()
create_dirs([config.summary_dir, config.checkpoint_dir])

# create your data generator
data = CompressedImageNetDataGenerator(config)
model = PrototypicalNetwork(config)

sess = tf.Session()
logger = Logger(sess, config)
trainer = ProtoNetTrainer(sess, model, data, config, logger)
model.load(sess)
trainer.train()

send_msg("Done")
