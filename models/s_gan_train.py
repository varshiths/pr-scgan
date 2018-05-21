import tensorflow as tf
from .base_train import BaseTrain


class SGANTrain(BaseTrain):

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):

            for k in range(self.config.disc_ascents):

                batch = self.data.random_batch()

                feed = {}
                feed[self.model.image.name] = batch["images"]
                feed[self.model.label.name] = batch["labels"]

                self.sess.run(self.model.disc_grad_step, feed)

            batch = self.data.next_batch()
            while batch is not None:

                feed = {}
                feed[self.model.image.name] = batch["images"]
                feed[self.model.label.name] = batch["labels"]

                batch = self.data.random_batch()
                self.sess.run(self.model.gen_grad_step, feed)

                batch = self.data.next_batch()

            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
