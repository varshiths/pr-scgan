import os
import csv
import numpy as np
from pyquaternion import Quaternion


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

def general_pad(x, target_length):

    pads = target_length-x.shape[0]
    if pads >= 0:
        return np.pad(x, [ (0,pads), (0,0) ], 'constant')
    else:
        return x[:target_length, :]

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

    z, x, y = a
    matrix = eulerAnglesToRotationMatrix((x,y,z))
    qrt = Quaternion(matrix=matrix)
    qrt = (int(qrt[0]>0)*2-1) * qrt

    return qrt[0], qrt[1], qrt[2], qrt[3]

def quart_to_euler(a):

    qrt = Quaternion(a)
    matrix = qrt.rotation_matrix
    x, y, z = rotationMatrixToEulerAngles(matrix)

    return z, x, y