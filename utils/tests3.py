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

def run_cseqgan_and_output_org_out(sess, model, data, config, dirname):

	batch = data.random_batch()

	feed = {
		model.sentence.name	: batch["annotations"],
		model.latent.name	: sess.run(model.latent_distribution_sample),
		model.start.name	: sess.run(model.start_token),
	}
	produced = run_with_input_and_denormalize(sess, model, data.denormalise, feed)
	original = data.denormalise(batch["gestures"])

	print("Writing CSEGAN Produced and Original to: %s" % dirname)
	create_dirs([dirname])
	for i in range(produced.shape[0]):
		np.savetxt("%s/gesture.%d.prod.csv" % (dirname, i), np.transpose(produced[i, :, :]), delimiter=",")
		np.savetxt("%s/gesture.%d.orgn.csv" % (dirname, i), np.transpose(original[i, :, :]), delimiter=",")
