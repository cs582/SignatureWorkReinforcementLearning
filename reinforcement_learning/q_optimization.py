import torch
import logging
import numpy as np

logger = logging.getLogger("ReinforcementLearning/q_optimization.py")


def optimize_dqn(dqn, target, experience_batch, loss_function, gamma, optimizer, device, model_name):
    logger.info(f"Called {model_name} Optimizer")

    logger.debug("Unpacking Batch")
    mask_non_terminal_states = torch.BoolTensor([(x[3] is not None) for x in experience_batch])

    curr_images = torch.Tensor(np.array([x[0][0].cpu().numpy() for x in experience_batch])).to(device=device).double()
    curr_actions = torch.Tensor(np.asanyarray([x[1] for x in experience_batch])).to(device=device).long()
    curr_rewards = torch.Tensor(np.asanyarray([[x[2]] for x in experience_batch])).to(device=device)
    next_state_images = torch.Tensor(np.array([x[3][0].cpu().numpy() for x in experience_batch if (x[3] is not None)])).to(device=device).double()
    logger.debug("Batch Unpacked")

    # Predict the next moves
    logger.debug("Predict next moves")
    y_hat = dqn(curr_images).gather(1, curr_actions)
    logger.debug(f"Next moves predicted: {y_hat}")

    # Calculate target value
    logger.debug("Calculate target input value")

    if model_name == "Dueling_DQN" or model_name == "Double_DQN":
        # Double Q-Learning
        model_actions = target(next_state_images).data.max(1)[1]
        model_actions = model_actions.view(1, len(experience_batch))

    target_output = torch.as_tensor(torch.zeros_like(torch.empty(len(experience_batch), y_hat.shape[1], device=device, dtype=torch.double), device=device, dtype=torch.double), dtype=torch.double, device=device)
    target_output_values = target(next_state_images)

    if model_name == "Dueling_DQN" or model_name == "Double_DQN":
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