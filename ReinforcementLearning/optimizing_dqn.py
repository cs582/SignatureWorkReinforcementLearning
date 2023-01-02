import torch


def optimize_dqn(dqn, target, experience_batch, loss_function, gamma, optimizer):
    mask_non_terminal_states = None
    try:
        # maps to true those values with a Terminal state
        mask_non_terminal_states = torch.BoolTensor([x[3] is not None for x in experience_batch])
    except:
        for x in experience_batch:
            print("failed in making mask")
            print("X[0]", x[0][0].numpy())
            print("X[1]", x[1])
            print("X[2]", x[2])
            print("X[3]", x[3][0].numpy())


    curr_images = None

    try:
        curr_images = torch.Tensor([x[0][0].numpy() for x in experience_batch])
    except:
        for x in experience_batch:
            print("failed in unpacking cur state")
            print("X[0]", x[0][0].numpy())
            print("X[1]", x[1])
            print("X[2]", x[2])
            print("X[3]", x[3][0].numpy())


    curr_actions = None
    try:
        curr_actions = torch.Tensor([x[1] for x in experience_batch])
    except:
        for x in experience_batch:
            print("failed in unpacking actions")
            print("X[0]", x[0][0].numpy())
            print("X[1]", x[1])
            print("X[2]", x[2])
            print("X[3]", x[3][0].numpy())

    curr_rewards = None
    try:
        curr_rewards = torch.Tensor([x[2] for x in experience_batch])
    except:
        for x in experience_batch:
            print("failed in unpacking rewards")
            print("X[0]", x[0][0].numpy())
            print("X[1]", x[1])
            print("X[2]", x[2])
            print("X[3]", x[3][0].numpy())

    next_state_images = None
    try:
        # Only preserve those next_states that are not terminal
        next_state_images = torch.Tensor([x[3][0].numpy() for x in experience_batch if x[3] is not None])
    except:
        for x in experience_batch:
            print("failed in unpacking next_state")
            print("X[0]", x[0][0].numpy())
            print("X[1]", x[1])
            print("X[2]", x[2])
            print("X[3]", x[3][0].numpy())

    # Predict the next moves
    dqn = dqn.double()
    y_hat = dqn(curr_images.double()).gather(1, curr_actions.long())

    # Calculate target value
    y_target = torch.as_tensor(torch.zeros_like(torch.empty(len(experience_batch), y_hat.shape[1]))).double()
    target = target.double()
    y_target[mask_non_terminal_states] = gamma*target(next_state_images.double()) + curr_rewards

    # Calculate Loss
    loss = loss_function(y_hat, y_target)
    optimizer.zero_grad()
    loss.backward()

    # Take gradient step
    optimizer.step()

    return loss.item()


