import tensorflow as tf
import numpy as np
import math

def np_min(arr, gamma=1.0):
    '''
    Implements soft min among a list of tensors x = [x1, x2, ...]
    of dimensions [b] each returns a [b] sized tensor
    '''
    arr = np.stack(arr)
    return -gamma * np.log( np.sum( np.exp(-arr/gamma), axis=0) )

def np_dtw(x1, x2):
    b , t1, s  = x1.shape
    b_, t2, s_ = x2.shape
    assert b == b_, "Batch sizes are not the same"

    r = np.zeros((b, t1+1, t2+1))
    r[:, 1:, :] = math.inf
    r[:, :, 1:] = math.inf
    r[:, 0, 0] = 0

    for i in range(1, t1+1):
        for j in range(1, t2+1):
            cost = np.linalg.norm(x1[:, i-1]-x2[:, j-1], axis=1)
            val = cost + hard_min([
                    r[:, i-1, j  ],
                    r[:, i  , j-1],
                    r[:, i-1, j-1]
                ], gamma=1.0)
            r[:, i, j] = val

    return r[:, t1, t2]

def soft_min(arr, gamma=1.0):
    '''
    Implements soft min among a list of tensors x = [x1, x2, ...]
    of dimensions [b] each returns a [b] sized tensor
    '''
    arr = tf.stack(arr)
    return -gamma * tf.log( tf.reduce_sum( tf.exp(-arr/gamma), axis=0) )

def soft_dtw(x1, x2):
    '''
    Implements Soft DTW loss between two tensors x1, x2
    of dimensions [b, t_x1, size] and [b, t_x2, size]
    '''
    b , t1, s  = x1.shape.as_list()
    b_, t2, s_ = x2.shape.as_list()
    assert b == b_, "Batch sizes are not the same"

    r = np.zeros((b, t1+1, t2+1))
    r[:, 1:, :] = math.inf
    r[:, :, 1:] = math.inf
    r[:, 0, 0] = 0.0
    r = tf.Variable(r, dtype=tf.float32)

    for i in range(1, t1+1):
        for j in range(1, t2+1):
            cost = tf.norm(x1[:, i-1]-x2[:, j-1], axis=1)
            val = cost + soft_min([
                    r[:, i-1, j  ],
                    r[:, i  , j-1],
                    r[:, i-1, j-1]
                ], gamma=1.0)
            indices = np.zeros((b, 3), dtype=np.int32)
            indices[:, 0] = np.arange(b)
            indices[:, 1] = i
            indices[:, 2] = j
            r = tf.scatter_nd_update(r, indices, val)

    return r[:, t1, t2]

if __name__ == '__main__':

    x1 = tf.random_normal((10,3,32))
    x2 = tf.random_normal((10,4,32))

    temp = soft_dtw(x1, x2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        val = temp.eval(session=sess)
        import pdb; pdb.set_trace()
        print()