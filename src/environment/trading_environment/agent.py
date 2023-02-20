import numpy as np
import random
from logs.logger_file import logger_main


class Agent:
    def __init__(self, n_tokens: int = 10, memory_size: int = 1000, min_epsilon: float = 1e-4, decay_rate: float = 0.99, n_classes: int = 8):
        """Initialize the Agent object.
        Args:
            n_tokens (int): Number of tokens in the portfolio.
            memory_size (int): Maximum size of the experience replay memory.
            min_epsilon (float): Minimum probability of choosing a random action.
            decay_rate (float): Rate of decay of the epsilon probability.
            n_classes (int): Number of classes in the environment
        """
        logger_main.info("Initializing Agent")
        self.n_tokens = n_tokens
        self.memory_size = memory_size
        self.memory = []
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.n_classes = n_classes
        self.action = None

    def store(self, info):
        """Store the current experience in the replay memory.
        Args:
            info (tuple): Tuple of (state, action, reward, next_state).
        """
        logger_main.debug("Storing new experience in replay memory")
        self.memory.append(info)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        logger_main.debug("Experience stored")

    def draw(self, batch_size):
        """Draw a random sample of experiences from the replay memory.
        Args:
            batch_size (int): Size of the sample to draw.
        Returns:
            list: List of experiences, where each experience is a tuple of (state, action, reward, next_state).
        """
        logger_main.debug("Drawing random sample of experiences from replay memory")
        sample = random.sample(self.memory, min(batch_size, len(self.memory)))
        logger_main.debug(f"Sample of size {len(sample)} drawn from replay memory")
        return sample

    def get_action(self, y_hat, epsilon, episode):
        """Choose an action to take based on the estimated Q-values from the DQN.
        Args:
            y_hat (torch.Tensor): Estimated Q-values from the DQN.
            epsilon (float): Probability of choosing a random action.
            episode (int): Current training episode.
        Returns:
            numpy.ndarray: Binary array representing the action to take on each token.
        """
        # Change Action Making to a discrete set of actions
        logger_main.debug("Choosing action based on estimated Q-values")
        epsilon = max(epsilon * (self.decay_rate ** episode), self.min_epsilon)
        if np.random.rand() < epsilon:
            # Choose random actions
            logger_main.debug(f"Choosing random action with epsilon {epsilon}")
            self.action = np.random.randint(0, self.n_classes)
        else:
            # Choose actions with the highest Q-values
            logger_main.debug("Choosing action with highest Q-values")
            self.action = y_hat.data.max(1)[1].item()
        logger_main.debug(f"Action selected: {self.action}")
        return self.action
