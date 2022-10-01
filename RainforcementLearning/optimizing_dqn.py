import torch


def optimize_dqn(dqn, target, experience_batch, loss_function, gamma, optimizer):
    # maps to true those values with a Terminal state
    mask_non_terminal_states = torch.Tensor([x[3] is not None for x in experience_batch])

    curr_images = torch.Tensor([x[0] for x in experience_batch])
    curr_actions = torch.Tensor([x[1] for x in experience_batch])
    curr_rewards = torch.Tensor([x[2] for x in experience_batch])

    # Only preserve those next_states that are not terminal
    next_state_images = torch.Tensor([x[3] for x in experience_batch if x[3] is not None])

    # Predict the next moves
    y_hat = dqn(curr_images).gather(1, curr_actions)
    # Calculate target value
    y_target = torch.as_tensor(torch.zeros(len(experience_batch)))
    y_target[mask_non_terminal_states] = gamma*dqn(next_state_images).max(1)[0] + curr_rewards

    # Calculate Loss
    loss = loss_function(y_hat, y_target)
    optimizer.zero_grad()
    loss.backward()

    # Take gradient step
    optimizer.step()


