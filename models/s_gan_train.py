import tensorflow as tf
from .base_train import BaseTrain


class SGANTrain(BaseTrain):

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            pass
            # self.train_epoch()
            # self.sess.run(self.model.increment_cur_epoch_tensor)

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
