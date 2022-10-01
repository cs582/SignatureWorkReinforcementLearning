import numpy as np
import pandas as pd
import random

class Agent:
    def __init__(self, n_transactions, n_tokens, sequence_length=100, memory_size=100):
        self.map_actions = [1]*(n_transactions//2) + [-1]*(n_transactions//2) + [0]*n_tokens
        self.memory_size = memory_size
        self.sequence_length = sequence_length
        self.memory = [(None, None, None, None)]*self.memory_size
        self.curr_index_memory = 0

    def store(self, info):
        self.memory[self.curr_index_memory] = info
        self.curr_index_memory += 1

    def draw(self, batch_size):
        return random.sample(self.memory, batch_size)





