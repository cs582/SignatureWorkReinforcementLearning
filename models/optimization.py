import torch
import logging
import numpy as np

logger = logging.getLogger("reinforcement_learning/optimization.py")


def unpack_batch(batch, device):
    curr_states, curr_actions, curr_rewards, next_states = zip(*batch)
    curr_states = torch.stack([x[0] for x in curr_states]).to(device=device).double()
    curr_actions = torch.tensor(curr_actions, device=device).view(-1, 1)
    curr_rewards = torch.tensor(curr_rewards, device=device).view(-1, 1)
    next_states = torch.stack([x[0] for x in next_states if x is not None]).to(device=device).double()
    is_not_terminal = torch.BoolTensor([x is not None for x in next_states])
    return curr_states, curr_actions, curr_rewards, next_states, is_not_terminal


def optimize_dqn(dqn, target, batch, loss_fn, gamma, optimizer, device):
    logger.info(f"Q-Learning Optimization")

    logger.debug("Unpacking experience Batch")
    curr_states, curr_actions, curr_rewards, next_states, is_not_terminal = unpack_batch(batch, device)
    logger.debug("Batch Unpacked")

    # Predict next Q-values
    logger.debug("Predict next moves")
    y_hat = dqn(curr_states).gather(1, curr_actions)
    logger.debug(f"Next moves predicted: {y_hat}")

    # Calculate target Q-values
    logger.debug("Calculating target Q-values")

    # Compute the best action for the next state
    actions = target(next_states).data.max(1)[1].view(-1, 1)

    # Initialize a tensor to hold the Q-values
    target_q_val = torch.as_tensor(torch.zeros_like(torch.empty(len(batch), y_hat.shape[1], device=device, dtype=torch.double), device=device, dtype=torch.double), dtype=torch.double, device=device)

    # Use the predicted Q-values to update the target Q-values only for non-terminal states
    target_q_val[is_not_terminal] = gamma * target(next_states).gather(1, actions)

    # Add the immediate rewards to the target Q-values
    y_target = torch.add(target_q_val, curr_rewards*(2*(target_q_val > 0.0).double() - 1))
    logger.debug(f"Target Q-values calculated: {y_target}")

    # Calculate Loss
    logger.debug("Calculate the loss")
    loss = loss_fn(y_hat, y_target)
    logger.debug("Loss has been calculated")

    # Compute gradient
    logger.debug("Calculate gradient")
    optimizer.zero_grad()
    loss.backward()
    logger.debug("Gradient computed")

    # Take gradient step
    logger.debug("Taking gradient step")
    optimizer.step()
    logger.debug("Step has been taken")

    return loss.item()