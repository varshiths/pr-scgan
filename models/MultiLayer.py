
import tensorflow as tf

class MultiLayer(tf.layers.Layer):
	"""Implements MultiLayer Object that applies a list of Layer objects"""
	def __init__(self, *args, **kwargs):
		super(MultiLayer, self).__init__()
		self.layers = []
	
	def add_layer(self, layer):
		self.layers.append(layer)

	def call(self, inputs):
		out = inputs
		for layer in self.layers:
			out = layer.apply(out)
		return out

	def compute_output_shape(self, input_shape):
		output_shape = input_shape	
		for layer in self.layers:
			output_shape = layer.compute_output_shape(output_shape)
		return output_shape