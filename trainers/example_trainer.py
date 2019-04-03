import numpy as np
from tqdm import tqdm

from base.base_train import BaseTrain


class ProtoNetTrainer2(BaseTrain):

    def __init__(self, sess, model, data, config, logger, verbose=True):
        super(ProtoNetTrainer2, self).__init__(sess, model, data, config, logger)
        self.verbose = verbose

    def train_epoch(self):
        loop = range(self.config.num_episode_per_epoch)
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        if self.verbose:
            print("[loss = {}, acc = {}]".format(loss, acc))
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        inputs, query, labels = next(self.data.next_batch())
        feed_dict = {self.model.x: inputs,
                     self.model.q: query,
                     self.model.y: labels}
        _, loss, acc = self.sess.run([self.model.train_op, self.model.loss, self.model.acc],
                                     feed_dict=feed_dict)
        return loss, acc


class ProtoNetTrainer(BaseTrain):

    def __init__(self, sess, model, data, config, logger, verbose=True):
        super(ProtoNetTrainer, self).__init__(sess, model, data, config, logger)
        self.verbose = verbose

    def train_epoch(self):
        loop = range(self.config.num_episode_per_epoch)
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        if self.verbose:
            print("[loss = {}, acc = {}]".format(loss, acc))
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        inputs, query, labels = next(self.data.next_batch())
        feed_dict = {self.model.inputs: inputs,
                     self.model.query: query,
                     self.model.labels: labels}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.acc],
                                     feed_dict=feed_dict)
        return loss, acc


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
