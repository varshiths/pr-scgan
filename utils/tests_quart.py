import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from .tests import *


def run_quart_model_and_plot_gesture(sess, model, data, config):

	print("Plotting Quart Variations")
	samples = np.random.randn(config.batch_size, config.latent_state_size)
	start = np.zeros((config.batch_size, config.sequence_width, 4))
	gesture = run_with_feed_and_denormalize(sess, model, data, samples, start)

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

def run_quart_model_and_output_csv(sess, model, data, config, dirname):

	print("Writing Quart Variations")
	samples = np.random.randn(config.batch_size, config.latent_state_size)
	start = np.zeros((config.batch_size, config.sequence_width, 4))
	gesture = run_with_feed_and_denormalize(sess, model, data, samples, start)

	create_dirs([dirname])

	for i in range(gesture.shape[0]):
		np.savetxt("%s/gesture.%d.csv" % (dirname, i), np.transpose(gesture[i, :, :]), delimiter=",")
