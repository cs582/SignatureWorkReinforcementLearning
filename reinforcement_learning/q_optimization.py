import torch
import logging
import numpy as np

logger = logging.getLogger("reinforcement_learning/q_optimization.py")


def unpack_experience_batch(experience_batch, device):
    curr_images = torch.zeros((len(experience_batch), *experience_batch[0][0].shape), device=device, dtype=torch.double)
    curr_actions = torch.zeros((len(experience_batch),), device=device, dtype=torch.long)
    curr_rewards = torch.zeros((len(experience_batch), 1), device=device, dtype=torch.double)
    next_state_images = torch.zeros((len(experience_batch), *experience_batch[0][3].shape), device=device, dtype=torch.double)
    mask_non_terminal_states = torch.zeros((len(experience_batch),), device=device, dtype=torch.bool)

    for i, exp in enumerate(experience_batch):
        curr_images[i] = exp[0]
        curr_actions[i] = exp[1]
        curr_rewards[i] = exp[2]
        if exp[3] is not None:
            next_state_images[i] = exp[3]
            mask_non_terminal_states[i] = 1
    return curr_images, curr_actions, curr_rewards, next_state_images, mask_non_terminal_states


def optimize_dqn(dqn, target, experience_batch, loss_function, gamma, optimizer, device, model_name):
    logger.info(f"Called {model_name} Optimizer")

    logger.debug("Unpacking Batch")
    curr_images, curr_actions, curr_rewards, next_state_images, mask_non_terminal_states = unpack_experience_batch(experience_batch, device)
    logger.debug("Batch Unpacked")

    # Predict the next moves
    logger.debug("Predict next moves")
    y_hat = dqn(curr_images).gather(1, curr_actions)
    logger.debug(f"Next moves predicted: {y_hat}")

    # Calculate target value
    logger.debug("Calculate target input value")

    if model_name in ["Dueling_DQN", "Double_DQN"]:
        # Double Q-Learning
        model_actions = target(next_state_images).data.max(1)[1]
        model_actions = model_actions.view(1, len(experience_batch))

    target_output = torch.as_tensor(torch.zeros_like(torch.empty(len(experience_batch), y_hat.shape[1], device=device, dtype=torch.double), device=device, dtype=torch.double), dtype=torch.double, device=device)
    target_output_values = target(next_state_images)

    if model_name in ["Dueling_DQN", "Double_DQN"]:
        # Double Q-Learning
        target_output[mask_non_terminal_states] = gamma*target_output_values.gather(1, model_actions)

    if model_name == "Single_DQN":
        # Single Stream Q-Learning
        target_output[mask_non_terminal_states] = gamma*target_output_values

    target_output = torch.add(target_output, curr_rewards)
    logger.debug(f"Target output has been calculated!!!: {target_output}")

    # Calculate Loss
    logger.debug("Calculate the loss")
    loss = loss_function(y_hat, target_output)
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