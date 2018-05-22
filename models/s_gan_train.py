import tensorflow as tf
from .base_train import BaseTrain
import pprint

pp = pprint.PrettyPrinter()


class SGANTrain(BaseTrain):

    def train(self):
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
                    self.model.image.name : batch["images"],
                    self.model.label.name : batch["labels"],
                }

                fetched = self.sess.run(fetches, feed)

                if k == 0:
                    print("^ Disc: %f \t Gen: %f" % (fetched["disc_cost"], fetched["gen_cost"]))

            for k in range(self.config.gen_descents):

                batch = self.data.random_batch()

                fetches = {
                    "train_step" : self.model.gen_grad_step,
                    "disc_cost" : self.model.disc_cost,
                    "gen_cost" : self.model.gen_cost,
                }

                feed = {
                    self.model.image.name : batch["images"],
                    self.model.label.name : batch["labels"],
                }

                fetched = self.sess.run(fetches, feed)

                if k == 0:
                    print(". Disc: %f \t Gen: %f" % (fetched["disc_cost"], fetched["gen_cost"]))

            # estimate validation accuracy
            self.estimate_val_accur()

    def estimate_val_accur(self):
        batch = self.data.validation_set()

        fetches = {
            "recons_error" : self.model.validation_error
        }

        feed = {
            self.model.image.name : batch["images"],
            self.model.label.name : batch["labels"],
        }

        fetched = self.sess.run(fetches, feed)

        pp.pprint(fetched)

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
