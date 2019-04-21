import tensorflow as tf

from configs.config import MiniImageNetConfig
from data_loader.data_generator import CompressedImageNetDataGenerator
from models.example_model import PrototypicalNetwork
from trainers.example_trainer import ProtoNetTrainer
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import send_msg

config = MiniImageNetConfig()
config.exp_name = "without_weight"
create_dirs([config.summary_dir, config.checkpoint_dir])

# create your data generator
data = CompressedImageNetDataGenerator(config)
model = PrototypicalNetwork(config, with_weight=False)

sess = tf.Session()
logger = Logger(sess, config)
trainer = ProtoNetTrainer(sess, model, data, config, logger)
model.load(sess)
trainer.train()

send_msg("Done")
