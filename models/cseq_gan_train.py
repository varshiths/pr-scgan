import tensorflow as tf
import numpy as np

from .gan_train import GANTrain

import pprint

pp = pprint.PrettyPrinter()


class CSeqGANTrain(GANTrain):

    def pre_train(self):

        for cur_epoch in range(self.config.num_pretrain_epochs):
            print("Pre-Train Epoch:", cur_epoch)
            self.pretrain_epoch(cur_epoch)
            self.model.igs += 1000 - self.model.gs%1000

    def pretrain_epoch(self, epoch_no):

        fetches = {
            "step" : self.model.gen_pretrain_grad_step,
            "cost" : self.model.gen_pretrain_cost,
            "norm" : self.model.gen_pretrain_grads,
        }
        if self.config.log and self.model.summary is not None:
            fetches["summary"] = self.model.summary

        batch, is_end = self.data.next_batch()
        i = 0
        while not is_end:
            i+=1
            feed = {
                self.model.gesture.name : batch["gestures"],
                self.model.sentence.name : batch["annotations"],
                self.model.latent.name : self.sess.run(self.model.latent_distribution_sample),
                self.model.start.name : self.sess.run(self.model.start_token),
            }
            fetched = self.sess.run(fetches, feed)
            print("%5d Cost: %f Norm: %f" % (i, fetched["cost"], fetched["norm"]))
            if self.config.log and self.model.summary is not None:
                self.model.writer.add_summary(fetched["summary"], self.model.gs + float(i)/10)

            self.model.igs()
            batch, is_end = self.data.next_batch()

    def disc_step(self):

        batch = self.data.random_batch()

        fetches = {
            "train_step" : self.model.disc_grad_step,
            "disc_cost" : self.model.disc_cost,
            "norm" : self.model.disc_grads,
            # "gen_cost" : self.model.gen_cost,
        }

        feed = {
            self.model.data.name : batch["data"],
            self.model.latent.name : self.sess.run(self.model.latent_distribution_sample),
            self.model.start.name : self.sess.run(self.model.start_token),
        }

        fetched = self.sess.run(fetches, feed)

        # print("^ Disc: %f \t Gen: %f" % (fetched["disc_cost"], fetched["gen_cost"]))
        print("^ Disc: %f Norm: %f" % (fetched["disc_cost"], fetched["norm"]))

    def gen_step(self):
        
        batch = self.data.random_batch()

        fetches = {
            "train_step" : self.model.gen_grad_step,
            # "disc_cost" : self.model.disc_cost,
            "gen_cost" : self.model.gen_cost,
            "norm" : self.model.gen_grads,
        }
        
        feed = {
            # self.model.data.name : batch["data"],
            self.model.latent.name : self.sess.run(self.model.latent_distribution_sample),
            self.model.start.name : self.sess.run(self.model.start_token),
        }

        fetched = self.sess.run(fetches, feed)

        # print(". Disc: %f \t Gen: %f" % (fetched["disc_cost"], fetched["gen_cost"]))
        print(". \t Gen: %f Norm: %f" % (fetched["gen_cost"], fetched["norm"]))

    def validation_metrics(self):
        pass