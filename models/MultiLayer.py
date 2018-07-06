
import tensorflow as tf

class MultiLayer(tf.layers.Layer):
	"""Implements MultiLayer Object that applies a list of Layer objects"""
	def __init__(self, *args, **kwargs):
		super(MultiLayer, self).__init__(*args, **kwargs)
		self.layers = []
	
	def add_layer(self, layer):
		self.layers.append(layer)

	def call(self, inputs):
		out = inputs
		for layer in self.layers:
			out = layer(out)
		return out
		