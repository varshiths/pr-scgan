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
		data_train = self.load_jsl_from_folder()

		print("Downsampling data...")
		data_train = data_train[:,::2,:]

		print("Normalising data...")
		# process data
		print(data_train.shape)

		data_train_min = np.min(data_train, (0,1))
		data_train_max = np.max(data_train, (0,1))

		self.scale = np.maximum(data_train_max - data_train_min, np.exp(-5))
		self.offset = data_train_min

		data_train = ((data_train - self.offset)/self.scale)*2 - 1

		self.data_train = data_train


	def load_jsl_from_folder(self):

		data_dir, target_length = self.config.data_dir, self.config.sequence_length
		files = [str(x) for x in os.listdir(data_dir) if x[-4:] == ".csv"]

		all_data = []
		for file in files[:self.config.nfiles]:
			ffile = os.path.join(data_dir, file)

			data = np.transpose(np.genfromtxt(ffile, delimiter=','))
			all_data.append(data)
			print("%s \t %s" % (file, data.shape))

		all_data = [ general_pad(x, target_length) for x in all_data ]
		all_data = np.stack(all_data, axis=0)

		return all_data
		
	def denormalise(self, data):

		ret = ((data+1)/2)*self.scale + self.offset
		return ret

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
		