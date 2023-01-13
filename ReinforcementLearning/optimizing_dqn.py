import torch
import logging
import numpy as np

logger = logging.getLogger("ReinforcementLearning -> optimizing_dqn")


def optimize_dqn(dqn, target, experience_batch, loss_function, gamma, optimizer, device):
    logger.info("Called DQN Optimizer")

    logger.debug("Creating mask tensor")
    mask_non_terminal_states = torch.BoolTensor([(x[3] is not None) for x in experience_batch])

    logger.debug("Creating curr_images tensor")
    curr_images = torch.Tensor(np.array([x[0][0].cpu().numpy() for x in experience_batch])).to(device=device).double()
    logger.debug("Creating curr_actions tensor")
    curr_actions = torch.Tensor(np.asanyarray([x[1] for x in experience_batch])).to(device=device).long()
    logger.debug("Creating curr_rewards tensor")
    curr_rewards = torch.Tensor(np.asanyarray([[x[2]] for x in experience_batch])).to(device=device)
    logger.debug("Creating next_state_images tensor")
    next_state_images = torch.Tensor(np.array([x[3][0].cpu().numpy() for x in experience_batch if (x[3] is not None)])).to(device=device).double()
    logger.info("Unpacked Batch")

    # Predict the next moves
    logger.debug("Predict next moves")
    y_hat = dqn(curr_images).gather(1, curr_actions)
    logger.debug("Next moves predicted")

    # Calculate target value
    logger.debug("Calculate target input value")
    target_output = torch.as_tensor(torch.zeros_like(torch.empty(len(experience_batch), y_hat.shape[1], device=device, dtype=torch.double), device=device, dtype=torch.double), dtype=torch.double, device=device)
    logger.debug(f"mask shape: {mask_non_terminal_states.shape}, next images shape: {next_state_images.shape}")
    try:
        target_output[mask_non_terminal_states] = torch.add(gamma*target(next_state_images), curr_rewards)
        logger.debug("Target output has been calculated!!!")
    except:
        for i in range(len(experience_batch)):
            logger.debug(f"index {i} is None? {mask_non_terminal_states[i]}")
            logger.debug(f"next_img is = {experience_batch[i][3]}")

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


