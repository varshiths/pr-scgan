import tensorflow as tf
import numpy as np
from .base_data import DataMode
from .base_data import BaseData

from tensorflow.examples.tutorials.mnist import input_data

from .utils import convert_to_one_hot


class MNIST(BaseData):
	"""MNIST dataset"""
	def __init__(self, config):
		super(MNIST, self).__init__(config)
		self.iter_train = 0
		self.iter_eval = 0

		# download/load if not already present
		mnist = input_data.read_data_sets("MNIST_data/")

		self.data_train = mnist.train.images # Returns np.array
		self.labels_train = convert_to_one_hot(np.asarray(mnist.train.labels, dtype=np.int32), (0, 9))

		self.data_eval = mnist.test.images # Returns np.array
		self.labels_eval = convert_to_one_hot(np.asarray(mnist.test.labels, dtype=np.int32), (0, 9))

	def next_batch(self, dataMode):

		batch = {}

		if dataMode == DataMode.TRAIN:

			batch["images"] = self.data_train[self.iter_train : self.iter_train + self.config.batch_size]
			batch["labels"] = self.labels_train[self.iter_train : self.iter_train + self.config.batch_size]

			self.iter_train += self.config.batch_size
			if self.iter_train > self.data_train.shape[0]:
				self.iter_train = 0

		elif dataMode == DataMode.TEST:

			batch["images"] = self.data_eval[self.iter_eval : self.iter_eval + self.config.batch_size]
			batch["labels"] = self.labels_eval[self.iter_eval : self.iter_eval + self.config.batch_size]

			self.iter_eval += self.config.batch_size
			if self.iter_eval > self.data_eval.shape[0]:
				self.iter_eval = 0

		return batch


		