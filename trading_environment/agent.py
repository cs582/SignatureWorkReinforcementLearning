import numpy as np
import logging

logger = logging.getLogger("trading_environment/agent")


class Agent:
    def __init__(self, n_transactions=10, n_tokens=20, memory_size=100):
        logger.info("Initializing Agent")

        self.n_transactions = n_transactions
        self.n_tokens = n_tokens

        self.memory_size = memory_size
        self.memory = np.asanyarray([(None, None, None, None)] * self.memory_size)
        self.curr_index_memory = 0

        self.actions = None

    def store(self, info):
        self.curr_index_memory %= self.memory_size

        logger.debug(f"Storing current experience at index {self.curr_index_memory}")
        self.memory[self.curr_index_memory] = info
        self.curr_index_memory += 1
        logger.debug(f"Experience successfully stored!!!")

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
            rand_action = np.zeros(self.n_tokens)
            rand_action[-self.n_transactions:] = 1
            np.random.shuffle(rand_action)

            self.actions = rand_action
            logger.debug(f"Chosen random actions with epsilon {epsilon}")
            logger.debug(f"actions to be performed: {self.actions}")
            return self.actions

        # Get the highest and lowest scores
        logger.debug(f"raw output from DQN: {y_hat}")
        sorted_indexes = y_hat.detach().cpu().argsort()[0]
        logger.debug(f"sorted indexes: {sorted_indexes}")
        self.actions = np.zeros(self.n_tokens)
        self.actions[sorted_indexes[-self.n_transactions:]] = 1
        logger.debug(f"actions to be performed: {self.actions}")

        return self.actions
