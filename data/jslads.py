import numpy as np
import os

from .jslad import JSLAD

from memory_profiler import profile
from .utils import *

import pprint; ppr = pprint.PrettyPrinter()
sprint = lambda x: ppr.pprint(x)

class JSLADS(JSLAD):
	"""JSLA dataset with discretized outputs with select joints data"""
	def get_data_path(self):
		return "JSLAD_data/data.npy"

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

		gestures = np.around((gestures+180.0)/self.config.dz_level).astype(int)

		gestures = gestures[10:20]

		# encode words into one hot encodings
		ann_encodings, lengths = self.process_input(sentences, indices_of_words)

		return [gestures, gesture_means, ann_encodings, lengths, indices_of_words]
		
	def denormalise(self, data_org):

		_shape = data_org.shape
		data = data_org.astype(np.float32) * 3 - 180.0
		
		pos = np.broadcast_to(self.gesture_means, (_shape[0], _shape[1], _shape[2], 3))
		ret = np.concatenate((pos, data), axis=-1)
		# reshape into format
		ret = np.reshape(ret, [_shape[0], _shape[1], -1])
		
		return ret