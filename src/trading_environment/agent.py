import numpy as np
import logging
import random

logger = logging.getLogger("trading_environment/agent")


class Agent:
    def __init__(self, n_transactions: int = 10, n_tokens: int = 10, memory_size: int = 1000):
        """Initialize the Agent object.
        Args:
            n_transactions (int): Number of transactions to make per day.
            n_tokens (int): Number of tokens in the portfolio.
            memory_size (int): Maximum size of the experience replay memory.
        """
        logger.info("Initializing Agent")
        self.n_transactions = n_transactions
        self.n_tokens = n_tokens
        self.memory_size = memory_size
        self.memory = []
        self.actions = None

    def store(self, info):
        """Store the current experience in the replay memory.
        Args:
            info (tuple): Tuple of (state, action, reward, next_state).
        """
        logger.debug("Storing new experience in replay memory")
        self.memory.append(info)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        logger.debug("Experience stored")

    def draw(self, batch_size):
        """Draw a random sample of experiences from the replay memory.
        Args:
            batch_size (int): Size of the sample to draw.
        Returns:
            list: List of experiences, where each experience is a tuple of (state, action, reward, next_state).
        """
        logger.debug("Drawing random sample of experiences from replay memory")
        sample = random.sample(self.memory, min(batch_size, len(self.memory)))
        logger.debug(f"Sample of size {len(sample)} drawn from replay memory")
        return sample

    def get_action(self, y_hat, epsilon):
        """Choose an action to take based on the estimated Q-values from the DQN.
        Args:
            y_hat (torch.Tensor): Estimated Q-values from the DQN.
            epsilon (float): Exploration rate, used to determine the probability of taking a random action.
        Returns:
            numpy.ndarray: Binary array representing the action to take on each token.
        """
        logger.debug("Choosing action based on estimated Q-values")
        if np.random.rand() < epsilon:
            # Choose random actions
            logger.debug(f"Choosing random action with epsilon {epsilon}")
            self.actions = np.zeros(self.n_tokens)
            self.actions[np.random.choice(self.n_tokens, self.n_transactions, replace=False)] = 1
        else:
            # Choose actions with the highest Q-values
            logger.debug("Choosing action with highest Q-values")
            self.actions = np.zeros(self.n_tokens)
            top_k_indices = y_hat.detach().cpu().argsort()[0][-self.n_transactions:]
            self.actions[top_k_indices] = 1
        return self.actions