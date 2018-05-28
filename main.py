
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import GAN, GANTrain, SGAN, SGANTrain, FF, FFTrain, SGANConv, SeqGAN, GANTrainPreTrain
from utils.config import process_config
from data import DataMode, MNIST

import numpy as np


flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("seed", None, "seed to ensure reproducibility")
flags.DEFINE_string("architecture", None, "architecture to use")

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

		tf.device("/device:GPU:0")

		if FLAGS.seed is not None:
			tf.set_random_seed(FLAGS.seed)
			np.random.seed(FLAGS.seed)

		# create model
		if FLAGS.architecture == "ff":
			model = FF(config)
		elif FLAGS.architecture == "gan":
			model = GAN(config)
		elif FLAGS.architecture == "sgan":
			model = SGAN(config)
		elif FLAGS.architecture == "seqgan":
			model = SeqGAN(config)
		elif FLAGS.architecture == "sganconv":
			model = SGANConv(config)

		with tf.Session() as session:

			if FLAGS.model is None:
				session.run(tf.global_variables_initializer())
			else:
				model.load(session, FLAGS.model)

			if FLAGS.mode == "train":
				if FLAGS.architecture == "ff":
					model_train = FFTrain(session, model, data, config, None)
				elif FLAGS.architecture == "gan":
					model_train = GANTrain(session, model, data, config, None)
				elif FLAGS.architecture == "seqgan":
					model_train = GANTrainPreTrain(session, model, data, config, None)
				else:
					model_train = SGANTrain(session, model, data, config, None)

				try:
					model_train.train()
				except KeyboardInterrupt as e:
					pass

				if FLAGS.save is not None:
					model.save(session, FLAGS.save)


if __name__ == '__main__':
	tf.app.run()