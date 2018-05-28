import tensorflow as tf
import numpy as np
from .base_data import DataMode
from .base_data import BaseData


class JSL(BaseData):
	"""MNIST dataset"""
	def __init__(self, config):
		super(JSL, self).__init__(config)
		self.iter_train = 0

		# download/load if not already present
		

		# generate for now
		self.data_train = np.random.randn(
			self.config.batch_size, 
			self.config.time_steps,
			self.config.sequence_width
			)

		self.labels_train = np.random.randn(
			self.config.batch_size
			)

		# process data

	def next_batch(self):

		if self.iter_train > self.data_train.shape[0]:
			self.iter_train = 0
			return None

		batch = {}

		batch["data"] = self.data_train[self.iter_train : self.iter_train + self.config.batch_size]
		batch["labels"] = self.labels_train[self.iter_train : self.iter_train + self.config.batch_size]

		self.iter_train += self.config.batch_size

		return batch

	def random_batch(self):

		batch = {}

		n_data = self.data_train.shape[0]
		choices = np.random.randint(0, n_data, [self.config.batch_size])

		batch["data"] = self.data_train[choices]
		batch["labels"] = self.labels_train[choices]

		return batch

	def training_set(self):

		batch = {}

		batch["data"] = self.data_train
		batch["labels"] = self.labels_train

		return batch

	def validation_set(self):

		batch = {}

		batch["data"] = self.data_eval
		batch["labels"] = self.labels_eval

		return batch
		