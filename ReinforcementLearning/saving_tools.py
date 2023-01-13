import torch


def save_model(model, episode, optimizer, train_history, PATH):
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_history,
    }, PATH)