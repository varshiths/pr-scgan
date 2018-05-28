import tensorflow as tf
from . import GANTrain
import pprint

pp = pprint.PrettyPrinter()


class GANTrainPreTrain(GANTrain):

    def train(self):
        self.pre_train()
        super(GANTrainPreTrain, self).train()

    def pre_train(self):

        print("Pre-trainer")
