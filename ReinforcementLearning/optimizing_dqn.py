import torch
import logging


def optimize_dqn(dqn, target, experience_batch, loss_function, gamma, optimizer, device):
    logging.info("Called DQN Optimizer")

    logging.debug("Creating mask tensor")
    mask_non_terminal_states = torch.BoolTensor([x[3] is not None for x in experience_batch])
    logging.debug("Creating curr_images tensor")
    curr_images = torch.Tensor([x[0][0].cpu().numpy() for x in experience_batch]).double().to(device=device)
    logging.debug("Creating curr_actions tensor")
    curr_actions = torch.Tensor([x[1] for x in experience_batch]).long().to(device=device)
    logging.debug("Creating curr_rewards tensor")
    curr_rewards = torch.Tensor([x[2] for x in experience_batch]).to(device=device)
    logging.debug("Creating next_state_images tensor")
    next_state_images = torch.Tensor([x[3][0].cpu().numpy() for x in experience_batch if x[3] is not None]).double().to(device=device)
    logging.info("Unpacked Batch")

    # Predict the next moves
    logging.debug("Predict next moves")
    y_hat = dqn(curr_images).gather(1, curr_actions)
    logging.debug("Next moves predicted")

    # Calculate target value
    logging.debug("Calculate target value")
    y_target = torch.as_tensor(torch.zeros_like(torch.empty(len(experience_batch), y_hat.shape[1], device=device, dtype=torch.double), device=device, dtype=torch.double), dtype=torch.double, device=device)
    y_target[mask_non_terminal_states] = gamma*target(next_state_images) + curr_rewards
    logging.debug("Target value has been calculated")

    # Calculate Loss
    logging.debug("Calculate the loss")
    loss = loss_function(y_hat, y_target)
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


