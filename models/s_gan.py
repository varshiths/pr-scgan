import tensorflow as tf
from .base_model import BaseModel


class SGAN(BaseModel):

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        pass