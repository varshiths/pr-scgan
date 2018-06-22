import tensorflow as tf
import numpy as np
import os
from .base_data import BaseData

from .utils import *


class JSL(BaseData):
	"""MNIST dataset"""
	def __init__(self, config):
		super(JSL, self).__init__(config)
		self.iter_train = 0
		self.data_path = "JSLQ_data/data.npy"

		print("Loading data...")
		# download/load if not already present
		
		npy_present, arrays = self.load_npy()

		if 	npy_present:

			print("Loading data from npy files...")
			self.data_train = arrays[0]
			self.data_means = arrays[1]

		else:

			data = self.load_jsl_from_folder()

			arrays = self.normalise(data)
			self.save_npy(arrays)

			self.data_train = arrays[0]
			self.data_means = arrays[1]

	def load_npy(self):
		arrays = []
		npy_present = False
		if os.path.exists(self.data_path):
			npy_present = True
			arrays = np.load(self.data_path)
		return npy_present, arrays

	def save_npy(self, arrays):

		print("Saving data to npy files...")
		np.save(self.data_path, arrays)

	def normalise(self, data):

		print("Downsampling data...")
		data_train = data[:,::,:]

		print("Transforming and selecting data...")
		# saving motion data for later

		_shape = data_train.shape
		data_train_shaped = np.reshape(data_train, (_shape[0], _shape[1], -1, 6))
		# sampling the angles alone
		data_means = data_train_shaped[0,0,:,:3]
		data_train = data_train_shaped[:,:,:,3:]

		# # converting them to quarternions
		# tshape = list(data_train.shape); tshape[-1] = 4

		data_train = np.swapaxes(data_train, 0, -1)
		data_train = euler_to_quart(data_train)
		data_train = np.swapaxes(data_train, 0, -1)

		return [data_train, data_means]
		
	def denormalise(self, data_org):

		_shape = data_org.shape
		data = np.swapaxes(data_org, 0, -1)
		data = quart_to_euler(data)
		data = np.swapaxes(data, 0, -1)
		
		pos = np.broadcast_to(self.data_means, (_shape[0], _shape[1], _shape[2], 3))
		ret = np.concatenate((pos, data), axis=-1)
		# reshape into format
		ret = np.reshape(ret, [_shape[0], _shape[1], -1])
		
		return ret

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
		