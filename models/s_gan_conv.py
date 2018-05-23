import tensorflow as tf
from .s_gan import SGAN
import pprint

pp = pprint.PrettyPrinter()


class SGANConv(SGAN):

    def __init__(self, arg ):
        super(SGANConv, self).__init__(arg)
        pp.pprint("Using Convolutional Model")

    def generator_network(self, state):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("weight", [self.config.input_size, self.config.output_size])
            b = tf.get_variable("bias", [self.config.output_size])

            out = tf.nn.softmax( tf.matmul(state, w) + b )

        return out

    def discriminator_network(self, inp):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("weight", [self.config.output_size + self.config.input_size, 1])
            b = tf.get_variable("bias", [1])

            prob = tf.nn.sigmoid( tf.matmul(inp, w) + b )

        return prob