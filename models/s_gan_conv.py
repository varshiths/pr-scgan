import tensorflow as tf
from .s_gan import SGAN
import pprint

pp = pprint.PrettyPrinter()

def conv_and_activate(inp, filt):
    return tf.nn.softplus(tf.nn.conv2d( inp, filt, strides=[1,1,1,1], padding="VALID" ))

class SGANConv(SGAN):

    def __init__(self, arg ):
        super(SGANConv, self).__init__(arg)
        pp.pprint("Using Convolutional Model")

    def generator_network(self, inp):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):

            c1w = tf.get_variable("c1w", [9, 9, 1, 3])
            c2w = tf.get_variable("c2w", [11, 11, 3, 5])
            c3w = tf.get_variable("c3w", [10, 10, 5, 10])

            inp = tf.reshape( inp, [-1, 28, 28, 1] )

            out = conv_and_activate( inp, c1w )
            out = conv_and_activate( out, c2w )
            out = conv_and_activate( out, c3w )

            out = tf.reshape(out, [-1, 10])

        return out

    def discriminator_network(self, x, y):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):

            c1w = tf.get_variable("c1w", [9, 9, 1, 3])
            c2w = tf.get_variable("c2w", [11, 11, 3, 5])
            c3w = tf.get_variable("c3w", [10, 10, 5, 10])

            inp = tf.reshape( x, [-1, 28, 28, 1] )
            out = conv_and_activate( inp, c1w )
            out = conv_and_activate( out, c2w )
            out = conv_and_activate( out, c3w )
            out = tf.reshape(out, [-1, 10])

            joint_state = tf.concat([out, y], axis=1)

            f1w = tf.get_variable("f1w", [20, 10])
            f1b = tf.get_variable("f1b", [10])
            f2w = tf.get_variable("f2w", [10, 1])
            f2b = tf.get_variable("f2b", [1])

            outf = tf.nn.sigmoid(tf.matmul( joint_state, f1w ) + f1b)
            outf = tf.nn.sigmoid(tf.matmul( outf, f2w ) + f2b)

        return outf
