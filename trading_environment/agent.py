import numpy as np
import logging

logger = logging.getLogger("Agent -> agent")


class Agent:
    def __init__(self, n_transactions=10, n_tokens=20, memory_size=100):
        logger.info("Initializing Agent")
        self.map_actions = [2] * (n_transactions // 2) + [0] * (n_transactions // 2) + [1] * n_tokens
        self.map_actions = np.asanyarray(self.map_actions)

        self.n_transactions = n_transactions
        self.n_tokens = n_tokens

        self.memory_size = memory_size
        self.memory = np.asanyarray([(None, None, None, None)] * self.memory_size)
        self.curr_index_memory = 0

        self.actions = None

    def store(self, info):
        logger.debug("Agent called store method")
        self.memory[self.curr_index_memory] = info
        self.curr_index_memory = (self.curr_index_memory % self.memory_size) + 1
        logger.debug(f"Stored info at memory index {self.curr_index_memory}")

    def draw(self, batch_size):
        logger.debug("Agent called draw method")

        actual_size = min(batch_size, self.curr_index_memory)
        logger.debug(f"Size of batch: {actual_size}")

        indexes = np.arange(0, self.curr_index_memory, dtype=np.int)
        np.random.shuffle(indexes)
        index_sample = indexes[:actual_size + 1]
        logger.debug(f"Random Sample: {indexes}")

        return self.memory[index_sample]

    def get_action(self, y_hat, epsilon):
        logger.debug("Agent called method get_action")
        if np.random.rand() < epsilon:
            logger.debug(f"Chosen random actions with epsilon {epsilon}")
            self.actions = np.random.shuffle(self.map_actions)
            return self.actions

        # Get the highest and lowest scores
        logger.debug(f"raw output from DQN: {y_hat}")
        sorted_indexes = y_hat.detach().cpu().argsort()[0]
        logger.debug(f"sorted indexes: {sorted_indexes}")
        self.actions = np.ones(self.n_tokens)
        self.actions[sorted_indexes[:self.n_transactions // 2]] = 0
        self.actions[sorted_indexes[-self.n_transactions // 2:]] = 2
        logger.debug(f"actions to be performed: {self.actions}")

        return self.actions
