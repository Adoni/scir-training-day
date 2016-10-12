import tensorflow as tf
import numpy
import copy

class mlp_postagger_model:
    def __init__(self, config):
        self.window_size=config['window_size']
        self.embedding_size=config['embedding_size']
