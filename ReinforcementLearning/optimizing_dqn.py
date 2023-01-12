import torch
import logging
import numpy as np


def optimize_dqn(dqn, target, experience_batch, loss_function, gamma, optimizer, device):
    logging.info("Called DQN Optimizer")

    logging.debug("Creating mask tensor")
    mask_non_terminal_states = torch.BoolTensor([x[3] is not None for x in experience_batch], device=device)
    logging.debug("Creating curr_images tensor")
    curr_images = torch.Tensor(np.array([x[0][0].cpu().numpy() for x in experience_batch])).to(device=device).double()
    logging.debug("Creating curr_actions tensor")
    curr_actions = torch.Tensor(np.array([x[1] for x in experience_batch])).to(device=device).long()
    logging.debug("Creating curr_rewards tensor")
    curr_rewards = torch.Tensor(np.array([[x[2]] for x in experience_batch])).to(device=device)
    logging.debug("Creating next_state_images tensor")
    next_state_images = torch.Tensor(np.array([x[3][0].cpu().numpy() for x in experience_batch if x[3] is not None])).to(device=device).double()
    logging.info("Unpacked Batch")

    # Predict the next moves
    logging.debug("Predict next moves")
    logging.debug(f"curr_images shape: {curr_images.shape}")
    logging.debug(f"curr_images = {curr_images}")
    y_hat = dqn(curr_images).gather(1, curr_actions)
    logging.debug("Next moves predicted")

    # Calculate target value
    logging.debug("Calculate target input value")
    target_raw_output = target(next_state_images)
    target_output = torch.as_tensor(torch.zeros_like(torch.empty(len(experience_batch), y_hat.shape[1], device=device, dtype=torch.double), device=device, dtype=torch.double), dtype=torch.double, device=device)
    logging.debug(f"output_raw_output = {target_raw_output.shape}")
    logging.debug(f"curr_reward = {curr_rewards.shape}")
    target_output[mask_non_terminal_states] = torch.add(gamma*target_raw_output, curr_rewards)
    logging.debug("Target output has been calculated!!!")

    # Calculate Loss
    logging.debug("Calculate the loss")
    loss = loss_function(y_hat, target_output)
    logging.debug("Loss has been calculated")

    # Compute gradient
    logging.debug("Calculate gradient")
    optimizer.zero_grad()
    loss.backward()
    logging.debug("Gradient computed")

    # Take gradient step
    logging.debug("Taking gradient step")
    optimizer.step()
    logging.debug("Step has been taken")

    return loss.item()


