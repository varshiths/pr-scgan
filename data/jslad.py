import numpy as np
import os

from .jsla import JSLA

from memory_profiler import profile
from .utils import *

import pprint; ppr = pprint.PrettyPrinter()
sprint = lambda x: ppr.pprint(x)

class JSLAD(JSLA):
	"""JSLA dataset with discretized outputs"""
	def get_data_path(self):
		return "JSLAD_data/data.npy"

	def normalise(self, data):

		gestures, sentences, indices_of_words = data

		print("Transforming and selecting data...")
		gst_lengths = np.array([ len(x) for x in gestures ])
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

		import pdb; pdb.set_trace()

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
		annotations, ann_lengths = self.process_input(sentences, indices_of_words)

		return [gesture_means, indices_of_words, gestures, gst_lengths, annotations, ann_lengths]
		
	def denormalise(self, data_org):

		_shape = data_org.shape
		data = data_org.astype(np.float32) * 3 - 180.0
		
		pos = np.broadcast_to(self.gesture_means, (_shape[0], _shape[1], _shape[2], 3))
		ret = np.concatenate((pos, data), axis=-1)
		# reshape into format
		ret = np.reshape(ret, [_shape[0], _shape[1], -1])
		
		return ret