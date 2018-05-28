import tensorflow as tf
from .base_train import BaseTrain
import pprint

pp = pprint.PrettyPrinter()


class SeqGANTrain(BaseTrain):

    def train(self):
        for cur_epoch in range(0, self.config.num_epochs + 1, 1):
            print("Epoch:", cur_epoch)

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
