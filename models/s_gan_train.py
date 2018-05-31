import tensorflow as tf
from .gan_train import GANTrain
import pprint

pp = pprint.PrettyPrinter()


class SGANTrain(GANTrain):

    def disc_step(self):

        batch = self.data.random_batch()

        fetches = {
            "train_step" : self.model.disc_grad_step,
            "disc_cost" : self.model.disc_cost,
            "gen_cost" : self.model.gen_cost,
        }

        feed = {
            self.model.image.name : batch["data"],
            self.model.label.name : batch["labels"],
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
            self.model.image.name : batch["data"],
            self.model.label.name : batch["labels"],
        }

        fetched = self.sess.run(fetches, feed)

        print(". Disc: %f \t Gen: %f" % (fetched["disc_cost"], fetched["gen_cost"]))

    def validation_metrics(self):
        batch = self.data.validation_set()

        fetches = {
            "recons_error" : self.model.validation_error
        }

        feed = {
            self.model.image.name : batch["data"],
            self.model.label.name : batch["labels"],
        }

        fetched = self.sess.run(fetches, feed)

        pp.pprint(fetched)


