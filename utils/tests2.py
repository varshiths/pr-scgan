import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from .tests import *

from data.utils import general_pad
from .utils import *


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
	

	gesture_jsl, _, _ = ljff_out = data.load_jsl_from_folder()
	_gestures = [ general_pad(x, config.sequence_length) for x in gesture_jsl ]
	gesture_org = np.stack(_gestures, axis=0)

	normd, _, _, _, _ = data.normalise(ljff_out)
	gesture_dns = data.denormalise(normd)
	gesture_dns = np.apply_along_axis(func1d=lambda x: conv_smooth(x), axis=1, arr=gesture_dns)

	np.savetxt("gesture_org.csv", np.transpose(gesture_org[0, :, :]), delimiter=",")
	np.savetxt("gesture_dns.csv", np.transpose(gesture_dns[0, :, :]), delimiter=",")

	gesture_org	 = np.reshape(gesture_org, [-1, config.sequence_length, 107, 6])
	gesture_dns	 = np.reshape(gesture_dns, [-1, config.sequence_length, 107, 6])
