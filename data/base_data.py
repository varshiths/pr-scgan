import numpy as np
from enum import Enum


class BaseData:
    def __init__(self, config):
        self.config = config

        print("Building dataset ...")

    def next_batch(self, batch_size):
        raise NotImplementedError
