import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def run_with_feed(sess, model, data, feed):
	
	fetches = {
		"out": model.out_gen.name,
	}

	feed = {
		model.latent.name: feed
	}

	fetched = sess.run(fetches, feed)

	out = data.denormalise(fetched["out"])
	
	return out

def run_model_and_plot_image(sess, model, data, config):

	def squarify(data):
		return np.reshape(data, (-1,28,28))

	samples = np.random.randn(25, config.latent_state_size)
	image = squarify(run_with_feed(sess, model, data, samples))

	print("Plotting Samples")
	fig=plt.figure(1, figsize=(28, 28))
	columns = 5
	rows = 5
	for i in range(1, columns*rows +1):
	    img = image[i-1,:,:]
	    fig.add_subplot(rows, columns, i)
	    plt.imshow(img, cmap="gray")
	    plt.colorbar()

	point1 = np.random.randn(config.latent_state_size)
	point2 = np.random.randn(config.latent_state_size)

	n_interp = 4

	direct_int = squarify(run_with_feed(sess, model, data, interpolate(point1, point2, n_interp)))
	spher_int = squarify(run_with_feed(sess, model, data, spher_interpolate(point1, point2, n_interp)))

	print("Plotting Interpolations")
	fig2=plt.figure(2)
	for i in range(1, n_interp + 1):
	    img = direct_int[i-1,:,:]
	    fig2.add_subplot(2, n_interp, i)
	    plt.imshow(img, cmap="gray")
	for i in range(n_interp + 1, 2 * n_interp + 1):
	    img = direct_int[i-1-n_interp,:,:]
	    fig2.add_subplot(2, n_interp, i)
	    plt.imshow(img, cmap="gray")

	plt.show()

def interpolate(point1, point2, n):

	diff = (point2 - point1)/(n)
	points = []
	for i in range(n+1):
		points.append(point1 + diff * i)

	return np.stack(points, axis=0)

def spher_interpolate(point1, point2, n):
	val = np.linspace(0, 1, n+1)
	points = []
	for i in range(n+1):
		points.append(slerp(val[i], point1, point2))

	return np.stack(points, axis=0)

def slerp(val, low, high):
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def run_model_and_plot_gesture(sess, model, data, config):

	# samples = np.random.randn(32, config.latent_state_size)
	# gesture = run_with_feed(sess, model, data, samples)

	# print("Plotting samples")
	# fig=plt.figure(1)
	# columns = 5
	# rows = 5
	# for i in range(1, columns*rows +1):
	#     img = gesture[i-1,:,:]
	#     fig.add_subplot(rows, columns, i)
	#     plt.imshow(img, cmap="gray")
	    # plt.colorbar()

	gesture = data.denormalise(data.random_batch()["data"])

	print("Plotting examples")
	fig=plt.figure(2)
	columns = 3
	rows = 1
	for i in range(1, columns*rows +1):
	    img = gesture[i-1,:,:]
	    fig.add_subplot(rows, columns, i)
	    plt.imshow(img, cmap="gray", aspect="auto")
	    # plt.colorbar()

	plt.show()

