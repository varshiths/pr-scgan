
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import SGAN
from utils.config import process_config
from data import DataMode, MNIST


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("config", None, "config file for hyper-parameters")
flags.DEFINE_string("dataset", "mnist", "dataset used for the model")
flags.DEFINE_string("mode", "test", "flag whether to train or test")

FLAGS = flags.FLAGS


def main(argv):

	# build config
	config = process_config(FLAGS.config)

	if FLAGS.dataset == "mnist":
		data = MNIST(config)

	# create model
	sgan = SGAN(config)

	# test or train
	if FLAGS.mode == "train":
		pass

if __name__ == '__main__':
	tf.app.run()