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
    model = PrototypicalNetwork(config)

    sess = tf.Session()
    logger = Logger(sess, config)
    trainer = ProtoNetTrainer(sess, model, data, config, logger)
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
    x_embedding, q_embedding, prototype = sess.run(fetches=[model.emb_x, model.emb_q, model.prototype],
                                                   feed_dict={model.x: train_inputs,
                                                              model.q: train_query,
                                                              model.y: train_labels})
    import numpy as np

    from sklearn.manifold import TSNE
    all = np.concatenate([x_embedding, q_embedding, prototype])
    tsne = TSNE(n_components=2, random_state=0)

    all_res = tsne.fit_transform(all)
    x_res, q_res, p_res = all[:len(x_embedding)], all[len(x_embedding):len(x_embedding) + len(q_embedding)], all[len(
        x_embedding) + len(q_embedding):]

    x_res = np.reshape(x_res, newshape=(20, 5, -1))
    q_res = np.reshape(q_res, newshape=(20, 15, -1))

    print("")

    pass


if __name__ == '__main__':

    tf.reset_default_graph()
    experiment = 'protoNet2'

    if experiment == 'protoNet2':
        run_proto_net()
    elif experiment == 'protoNet_embedding':
        generate_image_embedding()
    else:
        main()

    send_msg("train done")
    """
#%%
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

tf.reset_default_graph()
    
config = MiniImageNetConfig()
create_dirs([config.summary_dir, config.checkpoint_dir])

data = CompressedImageNetDataGenerator(config)
model = PrototypicalNetwork(config)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
model.load(sess)

train_inputs, train_query, train_labels = next(data.next_batch())
x_embedding, q_embedding, prototype = sess.run(fetches=[model.emb_x, model.emb_q, model.prototype],
                                               feed_dict={model.x: train_inputs,
                                                          model.q: train_query,
                                                          model.y: train_labels})
import numpy as np

from sklearn.manifold import TSNE
all = np.concatenate([x_embedding, q_embedding, prototype])
tsne = TSNE(n_components=2, random_state=0)

all_res = tsne.fit_transform(all)
x_res, q_res, p_res = all[:len(x_embedding)], all[len(x_embedding):len(x_embedding) + len(q_embedding)], all[len(
    x_embedding) + len(q_embedding):]

x_res = np.reshape(x_res, newshape=(20, 5, -1))
q_res = np.reshape(q_res, newshape=(20, 15, -1))

#%%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple'
c = 14
for s in range(5):
    plt.figure()
    plt.imshow(train_inputs[c][s])
    plt.grid()

plt.figure(figsize=(20,12))
plt.scatter(p_res[c][0],p_res[c][1], c='r',s=400)

plt.scatter(p_res[c-1][0],p_res[c-1][1],c='b',s=400)
for s in range(5):
    plt.scatter(x_res[c][s][0],x_res[c][s][1], c='r',alpha=0.8,s=100)
    
    plt.scatter(x_res[c-1][s][0],x_res[c-1][s][1],c='b', alpha=0.8,s=100)
    plt.text(x_res[c][s][0]+.03, x_res[c][s][1]+.03, "{}".format(s), fontsize=9)

#%%
        
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure(figsize=(20,12))

ax = fig.add_subplot(111, projection='3d')

n = 100

base = 6
for c in range(base,base+2):
    
    ax.scatter(p_res[c][0],p_res[c][1],p_res[c][2],marker=c,s=400)
    for s in range(5):
        
        ax.scatter(x_res[c][s][0], x_res[c][s][1], x_res[c][s][2], marker=c,c=colors[c], s = 100)
        #plt.scatter(x_res[c][s][0],x_res[c][s][1], c=colors[c], alpha=0.8, marker=c,s=100)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#%%

tf.reset_default_graph()
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=(5,1600))
singular_values,u,v = tf.svd(x)
sigma = tf.diag(singular_values)
s1,u1,v1, sigma1 = sess.run([singular_values,u,v,sigma],feed_dict={x:x_emb[0]})
uu, ss, vv = np.linalg.svd(x_emb[0])
    
    """
