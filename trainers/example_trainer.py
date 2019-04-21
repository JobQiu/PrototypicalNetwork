import numpy as np
from tqdm import tqdm

from base.base_train import BaseTrain


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

        val_losses = []
        val_accs = []

        for _ in range(self.config.num_episode_per_val_epoch):
            loss, acc = self.val_step()
            val_losses.append(loss)
            val_accs.append(acc)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        if self.verbose:
            print("[loss = {}, acc = {}, val_loss = {}, val_acc = {}]".format(loss, acc, val_loss, val_acc))
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

    def val_step(self):

        inputs, query, labels = next(self.data.next_val_batch())
        feed_dict = {self.model.x: inputs,
                     self.model.q: query,
                     self.model.y: labels}
        _, loss, acc = self.sess.run([self.model.train_op, self.model.loss, self.model.acc],
                                     feed_dict=feed_dict)
        return loss, acc

    def test_stpe(self):

        inputs, query, labels = next(self.data.next_batch())
        feed_dict = {self.model.x: inputs,
                     self.model.q: query,
                     self.model.y: labels}
        _, loss, acc = self.sess.run([self.model.train_op, self.model.loss, self.model.acc],
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
