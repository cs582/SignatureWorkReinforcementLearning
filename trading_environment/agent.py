import numpy as np
import logging


class Agent:
    def __init__(self, n_transactions=10, n_tokens=20, memory_size=100):
        logging.info("Initializing Agent")
        self.map_actions = [1] * (n_transactions // 2) + [-1] * (n_transactions // 2) + [0] * n_tokens
        self.map_actions = np.asanyarray(self.map_actions)

        self.n_transactions = n_transactions
        self.n_tokens = n_tokens

        self.memory_size = memory_size
        self.memory = np.asanyarray([(None, None, None, None)] * self.memory_size)
        self.curr_index_memory = 0

        self.actions = None

    def store(self, info):
        logging.info("Agent called store method")
        self.memory[self.curr_index_memory] = info
        self.curr_index_memory = (self.curr_index_memory % self.memory_size) + 1
        logging.info(f"\tstored info at {self.curr_index_memory}")

    def draw(self, batch_size):
        logging.info("Agent called draw method")

        actual_size = min(batch_size, self.curr_index_memory+1)
        logging.info(f"Size of batch = {actual_size}")

        indexes = np.arange(0, self.curr_index_memory, dtype=np.int)
        np.random.shuffle(indexes)
        index_sample = indexes[:actual_size]
        logging.info(f"Random Sample = {indexes}")

        return self.memory[index_sample]

    def get_action(self, y_hat, epsilon):
        logging.info("Agent called method get_action")
        if np.random.rand() < epsilon:
            logging.info(f"Chosen random actions with epsilon = {epsilon}")
            self.actions = np.random.shuffle(self.map_actions)
            return self.actions

        # Get the highest and lowest scores
        logging.info(f"raw output from DQN: {y_hat}")
        sorted_indexes = y_hat.detach().cpu().argsort()[0]
        logging.info(f"sorted indexes: {sorted_indexes}")
        self.actions = np.zeros(self.n_tokens)
        self.actions[sorted_indexes[:self.n_transactions // 2]] = -1
        self.actions[sorted_indexes[-self.n_transactions // 2:]] = 1
        logging.info(f"actions to be performed: {self.actions}")

        return self.actions
