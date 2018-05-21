import numpy as np
from enum import Enum


class DataMode(Enum):
	"""Indicates the mode of the data: train/test/val"""
	TRAIN = 0
	VAL = 1
	TEST = 2

class BaseData:
    def __init__(self, config):
        self.config = config

    def next_batch(self, batch_size):
        raise NotImplementedError
