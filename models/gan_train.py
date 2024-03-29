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

                self.disc_step()

            for k in range(self.config.gen_descents):

                self.gen_step()

        self.validation_metrics()

    def pre_train(self):

        print("Pre-Trainer")
        pass

    def disc_step(self):

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

    def gen_step(self):
        
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

    def validation_metrics(self):
        pass