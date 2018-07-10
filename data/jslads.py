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
		return "JSLADS_data/data.npy"

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

		# selecting the sections as per the config file
		jp = self.get_joint_indices_partition(self.config.sections)
		gestures_slct = gestures[:, :, jp[0], :]
		# gestures_nslct = gestures[0, 0, jp[1], :]
		gestures_nslct = np.mean(gestures[:, :, jp[1], :], axis=0,1)

		# encode words into one hot encodings
		ann_encodings, lengths = self.process_input(sentences, indices_of_words)

		return [gestures_slct, (gesture_means, gestures_nslct), ann_encodings, lengths, indices_of_words]
		
	def denormalise(self, data):

		# batch_size x seq_length x seq_width x 3
		batch_size, seq_length, seq_width, _ = data.shape

		jp = self.get_joint_indices_partition(self.config.sections)
		assert len(jp[0]) == seq_width

		# add the non select joints means 
		data_nslct = np.broadcast_to(self.gesture_means[1], (batch_size, seq_length, len(jp[1]), 3))
		data_slct = data

		data = np.zeros((batch_size, seq_length, 107, 3), dtype=np.float32)
		data[:, :, jp[0], :] = data_slct
		data[:, :, jp[1], :] = data_nslct

		# de discretize
		data = data.astype(np.float32) * 3 - 180.0
		
		# add the translation means
		pos = np.broadcast_to(self.gesture_means[0], (batch_size, seq_length, 107, 3))
		ret = np.concatenate((pos, data), axis=-1)
		# reshape into format
		ret = np.reshape(ret, [batch_size, seq_length, -1])
		
		return ret

	def get_joint_indices_partition(self, _sections):
		'''
		creates two lists of indices, one included in the sections
		and the complementary list
		'''
		# get mapping
		_map = self.section_indices_mapping()

		indices = [[], []]
		# always include hip
		if "base" not in _sections:
			_sections.append("base")

		for key, value in _map.items():
			if key in _sections:
				indices[0].extend(value)
			else:
				indices[1].extend(value)

		return indices

	def section_indices_mapping(self):
		# mapping keyword to the feature indices (all 6 orientation params grouped)
		# in the BVH files
		# notice the reshape (642) -> (107, 6)
		return {
			# hip
			"base": [0],
			"lower": list(range(1,8+1)),
			# chest + right arm + left arm
			"torso": list(range(9,11+1)) + list(range(12,16+1)) + list(range(36,40+1)),
			# right palm + left palm
			"fingers": list(range(17,35+1)) + list(range(41,59+1)),
			# above neck
			"face": list(range(60,106+1)),
		}