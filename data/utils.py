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

def general_pad(x, target_length):

	pads = target_length-x.shape[0]
	if pads >= 0:
		return np.pad(x, [ (0,pads), (0,0) ], 'constant')
	else:
		return x[:target_length, :]
