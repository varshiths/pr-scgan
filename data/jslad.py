import numpy as np
import os

from .jsla import JSLA

from memory_profiler import profile
from .utils import *

import pprint; ppr = pprint.PrettyPrinter()
sprint = lambda x: ppr.pprint(x)

class JSLAD(JSLA):
	"""JSLA dataset with discretized outputs"""
	def __init__(self, config):
		super(JSLA, self).__init__(config)
		self.iter_set = -1
		self.data_path = "JSLAD_data/data.npy"

		print("Loading data...")
		# download/load if not already present
		
		npy_present, storage = self.load_npy()

		if 	npy_present:
			print("Loading data from npy files...")
			storage = storage.item()
		else:
			print("Loading data from data directory...")
			data = self.load_jsl_from_folder()
			normalised = self.normalise(data); del data
			storage = self.split_into_test_train_eval(normalised); del normalised
			self.save_npy(storage)

		self.gesture_means = storage["gesture_means"]
		self.indices_of_words = storage["indices_of_words"]

		self.gestures = storage["gestures"]
		self.annotations = storage["annotations"]
		self.ann_lengths = storage["ann_lengths"]

		self.gestures_val = storage["gestures_val"]
		self.annotations_val = storage["annotations_val"]
		self.ann_lengths_val = storage["ann_lengths_val"]

		self.gestures_test = storage["gestures_test"]
		self.annotations_test = storage["annotations_test"]
		self.ann_lengths_test = storage["ann_lengths_test"]
		del storage

	def normalise(self, data):

		gestures, sentences, indices_of_words = data

		print("Transforming and selecting data...")
		_gestures = [ general_pad(x, self.config.sequence_length) for x in gestures ]
		gestures = np.stack(_gestures, axis=0); del _gestures

		# saving motion data for later
		_shape = gestures.shape
		gestures_shaped = np.reshape(gestures, (_shape[0], _shape[1], 107, 6))
		# sampling the angles alone
		gesture_means = gestures_shaped[0,0,:,:3]
		gestures = gestures_shaped[:,:,:,3:]

		# from tempfile import mkdtemp
		# filename = os.path.join(mkdtemp(), 'newfile.dat')
		# gmmap = np.memmap(filename, dtype='float32', shape=gestures.shape)
		# gmmap[:] = gestures[:]

		gestures = np.around((gestures+180.0)/self.config.dz_level).astype(int)
		# nclasses = int(360/self.config.dz_level)

		# all at once; causes OOM
		# gestures = convert_to_soft_one_hot(
		# 		gestures, 
		# 		nclasses, 
		# 		self.config.soft_label_window,
		# 		self.config.soft_label_dilution,
		# 	)

		# # split to avoid OOM
		# gestures_list = np.array_split(gestures, 16, axis=0); del gestures
		# fin_list = []
		# for gestures in gestures_list:
		# 	gestures1 = convert_to_soft_one_hot(
		# 			gestures, 
		# 			nclasses, 
		# 			self.config.soft_label_window,
		# 			self.config.soft_label_dilution,
		# 		)			
		# 	fin_list.append(gestures1)
		# gestures = np.concatenate(fin_list, axis=0); del fin_list

		# encode words into one hot encodings
		ann_encodings, lengths = self.process_input(sentences, indices_of_words)

		return [gestures, gesture_means, ann_encodings, lengths, indices_of_words]

	def process_input(self, sentences, indices_of_words=None):

		# obtaining dictionaries
		try:
			if indices_of_words is None:
				indices_of_words = self.indices_of_words
		except Exception as e:
			raise Exception("Dictionary not defined")

		nsentences = len(sentences)
		slen = self.config.annot_seq_length
		swidth = self.config.vocab_size

		ohencodings = np.zeros((nsentences, slen, swidth))
		lengths = np.zeros((nsentences), dtype=int)
		
		for i, sentence in enumerate(sentences):
			indices = [ indices_of_words[word] for word in sentence ]
			_len = len(sentence)
			lengths[i] = _len; poss = np.arange(_len)
			ohencodings[i, poss, indices] = 1.0

		return ohencodings, lengths
		
	def denormalise(self, data_org):

		_shape = data_org.shape
		data = np.swapaxes(data_org, 0, -1)
		data = quart_to_euler(data)
		data = np.swapaxes(data, 0, -1)
		
		pos = np.broadcast_to(self.gesture_means, (_shape[0], _shape[1], _shape[2], 3))
		ret = np.concatenate((pos, data), axis=-1)
		# reshape into format
		ret = np.reshape(ret, [_shape[0], _shape[1], -1])
		
		return ret