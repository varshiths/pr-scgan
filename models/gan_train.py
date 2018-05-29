import tensorflow as tf
import numpy as np
from .base_train import BaseTrain
import pprint

pp = pprint.PrettyPrinter()


class GANTrain(BaseTrain):

    def train(self):

        self.pre_train()

        for cur_epoch in range(0, self.config.num_epochs + 1, 1):

            print("Epoch:", cur_epoch)

            for k in range(self.config.disc_ascents):

                batch = self.data.random_batch()

                fetches = {
                    "train_step" : self.model.disc_grad_step,
                    "disc_cost" : self.model.disc_cost,
                    "gen_cost" : self.model.gen_cost,
                }

                feed = {
                    self.model.data.name : batch["data"],
                    self.model.latent.name : np.random.randn(self.config.batch_size, self.config.latent_state_size),
                }

                fetched = self.sess.run(fetches, feed)

                print("^ Disc: %f \t Gen: %f" % (fetched["disc_cost"], fetched["gen_cost"]))

            for k in range(self.config.gen_descents):

                batch = self.data.random_batch()

                fetches = {
                    "train_step" : self.model.gen_grad_step,
                    "disc_cost" : self.model.disc_cost,
                    "gen_cost" : self.model.gen_cost,
                }
                
                feed = {
                    self.model.data.name : batch["data"],
                    self.model.latent.name : np.random.randn(self.config.batch_size, self.config.latent_state_size),
                }

                fetched = self.sess.run(fetches, feed)

                print(". Disc: %f \t Gen: %f" % (fetched["disc_cost"], fetched["gen_cost"]))

    def pre_train(self):

        print("Pre-Trainer")
        pass

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
