import tensorflow as tf
import numpy as np
from .base_data import BaseData

from .utils import *


class JSL(BaseData):
	"""MNIST dataset"""
	def __init__(self, config):
		super(JSL, self).__init__(config)
		self.iter_train = 0

		print("Loading data...")
		# download/load if not already present
		data_train = load_jsl_from_folder(self.config.data_dir, self.config.time_steps)

		print("Normalising data...")
		# process data
		print(data_train.shape)

		self.data_train_mean = np.mean(data_train, (0,1))
		self.data_train_std = np.std(data_train, (0,1))
		data_train = (data_train - self.data_train_mean) / (self.data_train_std)

		self.data_train = data_train

	def next_batch(self):

		if self.iter_train > self.data_train.shape[0]:
			self.iter_train = 0
			return None

		batch = {}

		batch["data"] = self.data_train[self.iter_train : self.iter_train + self.config.batch_size]

		self.iter_train += self.config.batch_size

		return batch

	def random_batch(self):

		batch = {}

		n_data = self.data_train.shape[0]
		choices = np.random.randint(0, n_data, [self.config.batch_size])

		batch["data"] = self.data_train[choices]

		return batch

	def training_set(self):

		batch = {}

		batch["data"] = self.data_train

		return batch

	def validation_set(self):

		batch = {}

		batch["data"] = self.data_eval

		return batch
		