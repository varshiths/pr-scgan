import tensorflow as tf
from .base_train import BaseTrain
import pprint

pp = pprint.PrettyPrinter()


class FFTrain(BaseTrain):

    def train(self):
        for cur_epoch in range(0, self.config.num_epochs + 1, 1):

            print("Epoch:", cur_epoch)

            batch = self.data.next_batch()
            while batch is not None:

                fetches = {
                    "train_step" : self.model.train_step,
                    "cost" : self.model.cost,
                }

                feed = {
                    self.model.image.name : batch["images"],
                    self.model.label.name : batch["labels"],
                }

                fetched = self.sess.run(fetches, feed)

                # print("Cost: %f" % (fetched["cost"]))
                
                batch = self.data.next_batch()

            # estimate validation accuracy
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
