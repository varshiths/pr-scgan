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

	# gesture = data.denormalise(data.random_batch()["data"])

	create_dirs([dirname])

	for i in range(gesture.shape[0]):
		np.savetxt("%s/gesture.%d.csv" % (dirname, i), np.transpose(gesture[i, :, :]), delimiter=",")

def run_data_norm_and_denorm(sess, model, data, config):
	
	jsld = data.load_jsl_from_folder()
	jsld = np.reshape(jsld, [-1, 64, 107, 6])
	fno = 14
	jsld[0,:,:fno,3:] = 0
	jsld[0,:,fno+1:,3:] = 0

	jsld = np.reshape(jsld, [-1, 64, 107*6])

	np.savetxt("gesture_org.csv", np.transpose(jsld[0, :, :]), delimiter=",")
	
	dat, mean = data.normalise(jsld)
	data.data_means = mean

	jsld_nd = data.denormalise(dat)
	np.savetxt("gesture_nd.csv", np.transpose(jsld_nd[0, :, :]), delimiter=",")

	jsld	 = np.reshape(jsld, [-1, 64, 107, 6])
	jsld_nd	 = np.reshape(jsld_nd, [-1, 64, 107, 6])

def etc(sess, model, data, config):

	import pdb; pdb.set_trace()
	batch = data.random_batch()
