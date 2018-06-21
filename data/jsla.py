import tensorflow as tf
import numpy as np
import os
import codecs
import csv
from collections import OrderedDict
from .base_data import BaseData

from .utils import *


class JSLA(BaseData):
	"""MNIST dataset"""
	def __init__(self, config):
		super(JSLA, self).__init__(config)
		self.iter_set = -1
		self.data_path = "JSLA_data/data.npy"

		print("Loading data...")
		# download/load if not already present
		
		npy_present, arrays = self.load_npy()

		if 	npy_present:
			print("Loading data from npy files...")
		else:
			print("Loading data from data directory...")
			data = self.load_jsl_from_folder()
			arrays = self.normalise(data)
			self.save_npy(arrays)

		self.gestures = arrays[0]
		self.gesture_means = arrays[1]
		self.annotations = arrays[2]
		self.indices_of_words = arrays[3]

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

		gestures, sentences, indices_of_words = data

		print("Transforming and selecting data...")
		# saving motion data for later
		_shape = gestures.shape
		gestures_shaped = np.reshape(gestures, (_shape[0], _shape[1], -1, 6))
		# sampling the angles alone
		gesture_means = gestures_shaped[0,0,:,:3]
		gestures = gestures_shaped[:,:,:,3:]

		# convert to quaternion representations
		gestures = np.swapaxes(gestures, 0, -1)
		gestures = euler_to_quart(gestures)
		gestures = np.swapaxes(gestures, 0, -1)

		# encode words into one hot encodings
		ann_encodings = self.process_input(sentences, indices_of_words)

		return [gestures, gesture_means, ann_encodings, indices_of_words]

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

		outputs = np.zeros((nsentences, slen, swidth))
		
		for i, sentence in enumerate(sentences):
			indices = [ indices_of_words[word] for word in sentence ]
			poss = np.arange(len(sentence))
			outputs[i, poss, indices] = 1

		return outputs
		
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

	def load_jsl_from_folder(self):

		data_dir, target_length = self.config.data_dir, self.config.sequence_length

		# files = [str(x) for x in os.listdir(data_dir) if x[-4:] == ".csv"]
		with codecs.open(self.config.sentences_file, "r", encoding="shiftjis") as f:
			data = csv.reader(f, delimiter=",")
			files = [row[0] for row in data]

		if self.config.allfiles == -1:
			files = files[:4]
		nfiles = len(files)

		gestures = []
		for i, file in enumerate(files):
			ffile = os.path.join(data_dir, file)

			data = np.transpose(np.genfromtxt(ffile, delimiter=','))
			gestures.append(data)
			print("%4d/%d %s \t %s" % (i, nfiles, file, data.shape))

		gestures = [ inter_pad(x, target_length) for x in gestures ]
		gestures = np.stack(gestures, axis=0)

		# read the sentences from file
		sentences_all = {}
		with codecs.open(self.config.sentences_file, "r", encoding="shiftjis") as f:
			data = csv.reader(f, delimiter=",")
			for row in data:
				sentences_all[row[0]] = row[1:]
		sentences = []
		for file in files:
			sentences.append(sentences_all[file])

		# get the vocab dict of annotations
		indices_of_words = {}
		with codecs.open(self.config.words_file, "r", encoding="shiftjis") as f:
			data = csv.reader(f, delimiter=",")
			for row in data:
				indices_of_words[row[0]] = int(row[1])-1

		# remove special characters from dictionary
		del indices_of_words["入"]
		del indices_of_words["出"]
		# remove special characters from sentences
		sentences = [ x[1::3] for x in sentences ]

		return gestures, sentences, indices_of_words

	def next_batch(self):

		if self.iter_set == -1:
			# split into batches and store in a list 
			nsamples = self.gestures.shape[0] - self.gestures.shape[0] % self.config.batch_size
			if nsamples == 0:
				raise Exception("Not enough data to form a batch")
			nsamples = np.random.choice(np.arange(self.gestures.shape[0]), size=nsamples, replace=False)

			nsamples = np.reshape(nsamples, [-1, self.config.batch_size])

			self.batches_gestures = self.gestures[nsamples]
			self.batches_annotations = self.annotations[nsamples]
			self.iter_set = nsamples.shape[0]-1

		batch = {
			"gestures": self.batches_gestures[self.iter_set],
			"annotations": self.batches_annotations[self.iter_set]
		}
		self.iter_set -= 1
		return batch, self.iter_set == -1 # indicator to end

	def random_batch(self):

		n_data = self.gestures.shape[0]
		choices = np.random.randint(0, n_data, [self.config.batch_size])

		batch = {
			"gestures": self.gestures[choices],
			"annotations": self.annotations[choices]
		}
		return batch
