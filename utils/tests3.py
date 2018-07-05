import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from .dirs import *


def run_with_input_and_denormalize(sess, model, denorm_fn, feed):
	
	fetches = {
		"out": model.out_gen.name,
	}
	fetched = sess.run(fetches, feed)
	out = denorm_fn(fetched["out"])	
	return out

def run_cseqgan_and_interact(sess, model, data, config, dirname):

	batch = data.random_batch()

	feed = {
		model.sentence.name	: batch["annotations"],
		model.length.name	: batch["ann_lengths"],
		model.latent.name	: sess.run(model.latent_distribution_sample),
		model.start.name	: sess.run(model.start_token),
	}
	produced = run_with_input_and_denormalize(sess, model, data.denormalise, feed)
	original = data.denormalise(batch["gestures"])

	import pdb; pdb.set_trace()

	print("Test done")
	# create_dirs([dirname])
	# for i in range(produced.shape[0]):
	# 	np.savetxt("%s/gesture.%d.prod.csv" % (dirname, i), np.transpose(produced[i, :, :]), delimiter=",")
	# 	np.savetxt("%s/gesture.%d.orgn.csv" % (dirname, i), np.transpose(original[i, :, :]), delimiter=",")

def sigm(arr):
	return 1/(1+np.exp(-arr))

def tanh(arr):
	return 2*sigm(2*arr)-1

def norm(arr):
	return np.sqrt(np.sum(np.square(arr), axis=0, keepdims=True))

def normalize(arr):
	return arr / norm(arr)

def softmax(arr):
	return np.exp(arr) / np.sum(np.exp(arr), axis=0, keepdims=True)

def quart_actv(x, y):

	# present function
	# x = np.exp(x)
	# y = tanh(y)
	# nd = normalize(np.array([x, y]))

	# softmax
	# nd = softmax(np.array([x, y]))

	# # mod
	# x = np.exp(x)
	# y = np.exp(y)-np.exp(-y)
	# nd = normalize(np.array([x, y]))

	# mod
	x = np.log(1+np.exp(x))
	y = y
	nd = normalize(np.array([x, y]))


	return nd

def check_quart_actv(sess, model, data, config):

	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	# Make data.
	X = np.arange(-20, 20, 0.1)
	Y = np.arange(-20, 20, 0.1)
	X, Y = np.meshgrid(X, Y)
	# Z = sigm(X)
	# Z = sigm(Y)
	Zx, Zy = quart_actv(X, Y)

	# Plot the surface.
	surf = ax.plot_wireframe(X, Y, Zy)# cmap=cm.coolwarm,
	                       # linewidth=0, antialiased=False)
	surf = ax.plot_surface(X, Y, Zx, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)

	# Customize the z axis.
	ax.set_zlim(-2.01, 2.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()
