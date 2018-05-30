import os
import csv
import numpy as np


def num_elements(shape):
	num = 1
	for i in list(shape):
		num *= i
	return num

def range_len(rang):
	return rang[1]-rang[0]+1

def convert_to_one_hot(data, rang):

	data_shape = data.shape
	data_oned = data.reshape(num_elements(data_shape))

	enc_data_shape = ( num_elements(data_shape), range_len(rang) )
	enc_data = np.zeros(enc_data_shape)
	enc_data[np.arange(num_elements(data_shape)), data_oned] = 1

	enc_data_shape = list(data_shape)
	enc_data_shape.append(range_len(rang))
	enc_data = enc_data.reshape(enc_data_shape)

	return enc_data

def load_jsl_from_folder(data_dir, target_pad):
	files = [str(x) for x in os.listdir(data_dir) if x[-4:] == ".csv"]

	all_data = []
	for file in files[:2]:
		ffile = os.path.join(data_dir, file)

		data = np.transpose(np.genfromtxt(ffile, delimiter=','))
		all_data.append(data)
		print("%s \t %s" % (file, data.shape))

	all_data = [ np.pad(x, [ (0,target_pad-x.shape[0]), (0,0) ], 'constant') for x in all_data ]
	all_data = np.stack(all_data, axis=0)

	return all_data
