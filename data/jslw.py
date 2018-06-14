import tensorflow as tf
import numpy as np
import os
import sys
from .jsl import JSL

import codecs
from .utils import *


class JSLW(JSL):
	"""MNIST dataset"""
	def __init__(self, config):
		super(JSL, self).__init__(config)
		self.iter_train = 0
		self.data_path = "JSLQW_data/data.npy"

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

	def load_jsl_from_folder(self):

		data_dir, annot_dir, target_length = self.config.data_dir, self.config.annot_dir, self.config.sequence_length
		files = [str(x) for x in os.listdir(data_dir) if x[-4:] == ".csv"]

		all_data = []
		# phrase_count = []
		missing = []
		present = []
		for i, file in enumerate(files[:self.config.nfiles]):
			ffile = os.path.join(data_dir, file)
			anfile = os.path.join(annot_dir, file)

			data = np.transpose(np.genfromtxt(ffile, delimiter=','))

			try:
				with codecs.open(anfile, "r", "shift_jis") as anfileobj:
					bounds = np.genfromtxt(anfileobj, delimiter=',').astype(int)
				present.append(anfile)
			except FileNotFoundError as e:
				missing.append(anfile)
				continue

			# suntracting sync from frame indicators and picking data in range
			bounds = bounds[2:, 1:] - bounds[1][1]
			bounds = np.reshape(bounds, [-1, 6])[:, [0, -1]]
			temp = [ data[i:j] for i,j in bounds ]
			# phrase_count.append(len(temp))
			all_data.extend(temp)

			print("%d/%d : %s \t %s" % (i+1, self.config.nfiles, file, data.shape))

		all_data = [ inter_pad(x, target_length) for x in all_data ]
		all_data = np.stack(all_data, axis=0)

		return all_data
		