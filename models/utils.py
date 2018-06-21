import tensorflow as tf
import math

def soft_min(arr):
    '''
    Implements soft min among a list of tensors x = [x1, x2, ...]
    of dimensions [b] each returns a [b] sized tensor
    '''

    
        
    return arr[0]

def soft_dtw(x1, x2):
    '''
    Implements Soft DTW loss between two tensors x1, x2
    of dimensions [b, t_x1, size] and [b, t_x2, size]
    '''
    b , t1, s  = x1.shape.as_list()
    b_, t2, s_ = x2.shape.as_list()
    assert b == b_, "Batch sizes are not the same"

    r = tf.zeros((b, t1+1, t2+1))
    r[:, 1:, :] = math.inf
    r[:, :, 1:] = math.inf

    for i in range(1, t1+1):
        for j in range(1, t2+1):
            cost = tf.norm(x1[:, i]-x2[:, j], axis=1)
            r[:, i, j] = cost + soft_min([
                    r[:, i-1, j  ],
                    r[:, i  , j-1],
                    r[:, i-1, j-1]
                ])

    return r[:, t1, t2]
