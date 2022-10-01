import torch.nn as nn
import torch

from models import DQN
from models.optimizing_dqn import optimize_dqn
from agent.agent import Agent
from environment.trading_environment import Environment

if __name__ == "__main__":
    loss_function = nn.MSELoss()
    episodes = 100

    n_tokens = 10

    mini_batch_size = 20

    memory_size = 10000
    trading_days = 900

    epsilon = 1e-6
    gamma = 1e-3

    done = False

    # Initialize replay memory D to capacity N
    agent = Agent(memory_size=memory_size)
    environment = Environment()
    # Initialize action-value function Q with random weights
    q = DQN(action_size=10)

    optimizer = torch.optim.SGD(q.parameters(), lr=1e-4, momentum=0.9)

    for episode in range(0, episodes):
        _, cur_state, _ = environment.trade()
        for i in range(0, trading_days):
            # Initialize gradient to zero
            optimizer.zero_grad()

            # Predict or randomly choose an action
            y_hat = q(cur_state)
            cur_action = agent.get_action(y_hat, epsilon)

            # Trade portfolio with the given instructions
            cur_reward, next_image, done = environment.trade(cur_action)

            # Store experience in memory
            cur_experience = (cur_state, cur_action, cur_reward, next_image)
            agent.store(cur_experience)

            # Update current state
            cur_state = next_image

            # Get a random minibatch of transitions from memory
            experience_batch = agent.draw(batch_size=mini_batch_size)

            optimize_dqn(q, experience_batch, loss_function, gamma, optimizer)
