import os
import csv
import numpy as np
from pyquaternion import Quaternion
import multiprocessing


def num_elements(shape):
    num = 1
    for i in list(shape):
        num *= i
    return num

def range_len(rang):
    return rang[1]-rang[0]+1

def convert_to_one_hot(data, rang):

    data_shape = data.shape
    data_oned = data.reshape(num_elements(data_shape))

    enc_data_shape = ( num_elements(data_shape), range_len(rang) )
    enc_data = np.zeros(enc_data_shape)
    enc_data[np.arange(num_elements(data_shape)), data_oned] = 1

    enc_data_shape = list(data_shape)
    enc_data_shape.append(range_len(rang))
    enc_data = enc_data.reshape(enc_data_shape)

    return enc_data

def general_pad(x, tgt_len):

    pads = tgt_len-x.shape[0]
    pads_2 = int(pads/2)
    if pads >= 0:
        return np.pad(x, [ (pads_2,pads-pads_2), (0,0) ], 'edge')
    else:
        nds = np.sort(np.random.choice(np.arange(x.shape[0]), tgt_len, replace=False))
        return x[nds]

def inter_pad(x, tgt_len):

    length = x.shape[0]
    if length >= tgt_len:
        # downsample
        nds = np.sort(np.random.choice(np.arange(length), tgt_len, replace=False))
        return x[nds]
    elif 2 * length < tgt_len:
        # upsample and pad
        nds = np.sort(np.random.choice(np.arange(length), length+int((tgt_len-length)/2), replace=True))
        x = x[nds]
        pads = tgt_len-x.shape[0]; pads_2 = int(pads/2)
        return np.pad(x, [ (pads_2,pads-pads_2), (0,0) ], 'edge')
    else:
        # upsample
        nds = np.sort(np.random.choice(np.arange(length), tgt_len, replace=True))
        return x[nds]

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def euler_to_quart(a):
    '''
    convert degrees to rad and compute the relevant quaternion
    first axis corresponds to angles
    '''
    a = np.deg2rad(a).astype(np.float64)

    r, p, y = a

    # matrix = eulerAnglesToRotationMatrix((x,y,z))
    # qrt = Quaternion(matrix=matrix)

    cr = np.cos(r/2); sr = np.sin(r/2)
    cp = np.cos(p/2); sp = np.sin(p/2)
    cy = np.cos(y/2); sy = np.sin(y/2)

    q0 = cy*cr*cp + sy*sr*sp
    q1 = cy*sr*cp - sy*cr*sp
    q2 = cy*cr*sp + sy*sr*cp
    q3 = sy*cr*cp - cy*sr*sp


    qrt = np.stack([q0, q1, q2, q3], axis=0)    
    qrt = ((qrt[0]>0).astype(int)*2-1) * qrt

    return qrt.astype(np.float32)

def quart_to_euler(qrt):
    '''
    convert quart into euler in degrees
    first axis corresponds to quaternion components
    '''
    qrt = qrt.astype(np.float64)
    # qrt = Quaternion(a)

    # matrix = qrt.rotation_matrix
    # x, y, z = rotationMatrixToEulerAngles(matrix)

    q0, q1, q2, q3 = qrt[0], qrt[1], qrt[2], qrt[3]
    r = np.arctan2  ( 2*(q0*q1 + q2*q3) , 1-2*(q1**2+q2**2) )
    p = np.arcsin   ( np.clip(2*(q0*q2-q3*q1), -1.0, +1.0) )
    y = np.arctan2  ( 2*(q0*q3 + q1*q2) , 1-2*(q2**2+q3**2) )
    z, x, y = r, p, y

    a = np.stack([z, x, y], axis=0)

    a = np.rad2deg(a).astype(np.float32)
    return a

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    try:
        effective_axis = 1 if axis == 0 else axis
        if effective_axis != axis:
            arr = arr.swapaxes(axis, effective_axis)

        # Chunks for the mapping (only a few chunks):
        chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
                  for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

        pool = multiprocessing.Pool()
        individual_results = pool.map(unpacking_apply_along_axis, chunks)
        # Freeing the workers:
        pool.close()
        pool.join()
        return np.concatenate(individual_results)
    except Exception as e:
        print("Error: Pooling failed. Using single cpu. ~8 times longer")
        return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def unpacking_apply_along_axis(arg):
    func1d, axis, arr, args, kwargs = arg
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
