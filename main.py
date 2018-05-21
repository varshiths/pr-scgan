
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import SGAN, SGANTrain
from utils.config import process_config
from data import DataMode, MNIST


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("config", None, "config file for hyper-parameters")
flags.DEFINE_string("dataset", "mnist", "dataset used for the model")
flags.DEFINE_string("mode", "test", "flag whether to train or test")
flags.DEFINE_string("model", None, "loading saved model")
flags.DEFINE_string("save", None, "name of the model to save")

FLAGS = flags.FLAGS


def main(argv):

	# build config
	config = process_config(FLAGS.config)

	if FLAGS.dataset == "mnist":
		data = MNIST(config)

	with tf.Graph().as_default():
		# create model
		sgan = SGAN(config)
		sgan.init_saver()
		sgan.build_model()

		with tf.Session() as session:
			session.run(tf.global_variables_initializer())

			if FLAGS.model is not None:
				sgan.load(session, FLAGS.model)

			if FLAGS.mode == "train":
				sgan_train = SGANTrain(session, sgan, data, config, None)

				sgan_train.train()

				if FLAGS.save is not None:
					sgan.save(session, FLAGS.save)


if __name__ == '__main__':
	tf.app.run()