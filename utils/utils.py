import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

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