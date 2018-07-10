import tensorflow as tf
import numpy as np
import pandas as pd
import os
import codecs
import csv
from collections import OrderedDict
from .base_data import BaseData

from utils.dirs import create_dirs

import unicodedata as ucd

from memory_profiler import profile
from .utils import *

import pprint; ppr = pprint.PrettyPrinter()
sprint = lambda x: ppr.pprint(x)

class JSLA(BaseData):
	"""MNIST dataset"""
	def __init__(self, config):
		super(JSLA, self).__init__(config)
		self.iter_set = -1
		self.data_path = self.get_data_path()

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
		self.gst_lengths = storage["gst_lengths"]
		self.annotations = storage["annotations"]
		self.ann_lengths = storage["ann_lengths"]

		self.gestures_val = storage["gestures_val"]
		self.gst_lengths_val = storage["gst_lengths_val"]
		self.annotations_val = storage["annotations_val"]
		self.ann_lengths_val = storage["ann_lengths_val"]

		self.gestures_test = storage["gestures_test"]
		self.gst_lengths_test = storage["gst_lengths_test"]
		self.annotations_test = storage["annotations_test"]
		self.ann_lengths_test = storage["ann_lengths_test"]
		del storage

	def get_data_path(self):
		return "JSLA_data/data.npy"

	def load_npy(self):
		arrays = []
		npy_present = False
		if os.path.exists(self.data_path):
			npy_present = True
			arrays = np.load(self.data_path)
		return npy_present, arrays

	def save_npy(self, arrays):

		print("Saving data to npy files...")
		create_dirs([os.path.dirname(self.data_path)])
		np.save(self.data_path, arrays)

	def split_into_test_train_eval(self, normalized_data, seed=0):
		gesture_means, indices_of_words, gestures, gst_lengths, annotations, ann_lengths = normalized_data

		# shuffle
		ndata = gestures.shape[0]
		split = [int(ndata*0.8), int(ndata*0.9)]
		nindices = np.arange(ndata)
		np.random.shuffle(nindices)
		ntrain, nval, ntest  = np.split(nindices, split)
		# test train eval split
		split_data = {
			"gesture_means" 	: gesture_means,
			"indices_of_words" 	: indices_of_words,

			"gestures" 			: gestures[ntrain],
			"gst_lengths" 		: gst_lengths[ntrain],
			"annotations" 		: annotations[ntrain],
			"ann_lengths" 		: ann_lengths[ntrain],

			"gestures_val" 		: gestures[nval],
			"gst_lengths_val" 	: gst_lengths[nval],
			"annotations_val" 	: annotations[nval],
			"ann_lengths_val" 	: ann_lengths[nval],

			"gestures_test" 	: gestures[ntest],
			"gst_lengths_test" 	: gst_lengths[ntest],
			"annotations_test" 	: annotations[ntest],
			"ann_lengths_test" 	: ann_lengths[ntest],
		}
		return split_data

	def normalise(self, data):

		gestures, sentences, indices_of_words = data

		print("Transforming and selecting data...")
		gst_lengths = np.array([ len(x) for x in gestures ])
		gestures = [ general_pad(x, self.config.sequence_length) for x in gestures ]
		gestures = np.stack(gestures, axis=0)

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

		# # split to avoid OOM
		gestures_list = np.array_split(gestures, 16, axis=0); del gestures
		fin_list = []
		for gestures in gestures_list:
			# convert to quaternion representations
			gestures1 = np.swapaxes(gestures, 0, -1); del gestures
			gestures2 = euler_to_quart(gestures1); del gestures1
			gestures3 = np.swapaxes(gestures2, 0, -1); del gestures2
			fin_list.append(gestures3); del gestures3
		gestures = np.concatenate(fin_list, axis=0)

		# gestures = np.swapaxes(gestures, 0, -1)
		# gestures = euler_to_quart(gestures)
		# gestures = np.swapaxes(gestures, 0, -1)

		# encode words into one hot encodings
		annotations, ann_lengths = self.process_input(sentences, indices_of_words)

		return [gesture_means, indices_of_words, gestures, gst_lengths, annotations, ann_lengths]

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

	def load_jsl_from_folder(self):

		ofiles = [str(x) for x in os.listdir(self.config.data_dir) if x[-4:] == ".csv"]
		# with codecs.open(self.config.sentences_file, "r", encoding="shiftjis") as f:
		# 	data = csv.reader(f, delimiter=",")
		# 	files = [row[0] for row in data]
		files = ofiles[:20] if self.config.allfiles == -1 else ofiles
		nfiles = len(files)

		print("Process gestures")
		gestures = []
		for i, file in enumerate(files):
			ffile = os.path.join(self.config.data_dir, file)

			data = np.transpose(np.genfromtxt(ffile, delimiter=','))
			gestures.append(data)
			print("%4d/%d %s \t %s" % (i, nfiles, file, data.shape))

		'''
		# read the sentences from file
		headers = [	[ "FileName Rep. 1" , "手話単語"],
					[ "FileName Rep. 2" , "手話単語"],
					[ "FileName Rep. 3" , "手話単語"] ]
		arrays = []
		for file in self.config.sentences_files:
			# process strings to fill null values
			df = pd.read_excel(file, sheet_name=0).fillna("")
			theaders = headers[:-1] if file[file.rfind("/")+1:][:4] == "2018" else headers
			for head in theaders:
				arrays.append(df[head].values)
		data = np.concatenate(arrays)
		# remove empty cells
		data = data[np.where( np.all(data != "", axis=1) )]
		# process to replace special character "Ｂ" by "B"
		def process_name(element):
			# tp = element
			element[0] = element[0].replace("Ｂ", "B")
			return element
		data = np.apply_along_axis(
			func1d=process_name, 
			axis=1,
			arr=data)
		# sort in order of `files` after removing the `_t` (number indicating the take)
		# and scrap the names
		'''
		def _process_word(word):
			# normalize the full width characters into half width characters
			word = ucd.normalize("NFKC", word)
			# remove spaces between words and `(properties)`
			word = word.replace(" (", "(")
			word = word.replace(") ", ")")
			# remove ? ?
			# word = word.replace("?", "")
			# remove (Gen.) and split
			words = word.split("(Gen.)")
			# filter empty words
			words = [word for word in words if word != ""]
			return words

		def _process_sentence(sentence):
			temp = []
			for word in sentence:
				temp.extend(_process_word(word))
			return temp

		print("Process sentences")
		sentences = []
		for i, file in enumerate(ofiles):
			ffile = os.path.join(self.config.annot_dir, file)
			with codecs.open(ffile, "r", encoding="shiftjis") as f:
				data = csv.reader(f, delimiter=",")
				sentence = np.array(list(data))[3::3, 0].tolist()
			# process and split words
			# add an <eos> token
			sentence = _process_sentence(sentence)
			sentence.append("<eos>")

			# add to list
			sentences.append(sentence)

		words = {}
		for sentence in sentences:
			for word in sentence:
				if word not in words.keys():
					words[word] = 0
				words[word] += 1
		words_list = list(words.keys())
		words_list.sort()

		indices_of_words = { word: i for i, word in enumerate(words_list) }

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
			self.batches_gst_lengths = self.gst_lengths[nsamples]
			self.batches_annotations = self.annotations[nsamples]
			self.batches_ann_lengths = self.ann_lengths[nsamples]
			self.iter_set = nsamples.shape[0]-1

		batch = {
			"gestures": self.batches_gestures[self.iter_set],
			"gst_lengths": self.batches_gst_lengths[self.iter_set],
			"annotations": self.batches_annotations[self.iter_set],
			"ann_lengths": self.batches_ann_lengths[self.iter_set],
		}
		self.iter_set -= 1
		return batch, self.iter_set == -1 # indicator to end

	def random_batch(self):

		n_data = self.gestures.shape[0]
		choices = np.random.randint(0, n_data, [self.config.batch_size])

		batch = {
			"gestures": self.gestures[choices],
			"gst_lengths": self.gst_lengths[choices],
			"annotations": self.annotations[choices],
			"ann_lengths": self.ann_lengths[choices],
		}
		return batch

	def validation_batches(self):
		raise NotImplementedError

	def test_batches(self):
		raise NotImplementedError
