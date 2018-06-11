import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from .tests import *


def run_quart_model_and_plot_gesture(sess, model, data, config):

	samples = np.random.randn(config.batch_size, config.latent_state_size)
	start = np.zeros((config.batch_size, config.sequence_width))
	gesture = data.denormalise(run_with_feed(sess, model, data, samples, start))

	print("Plotting Quart Variations")
	fig=plt.figure(1)
	columns = 3
	rows = 1
	for i in range(1, columns*rows +1):
	    img = gesture[i-1,:,:]
	    fig.add_subplot(rows, columns, i)
	    plt.imshow(img, cmap="gray", aspect="auto")
	    plt.colorbar()

	# gesture = data.denormalise(data.random_batch()["data"])

	# print("Plotting examples")
	# fig=plt.figure(2)
	# columns = 3
	# rows = 1
	# for i in range(1, columns*rows +1):
	#     img = gesture[i-1,:,:]
	#     fig.add_subplot(rows, columns, i)
	#     plt.imshow(img, cmap="gray", aspect="auto")
	#     plt.colorbar()

	plt.show()
