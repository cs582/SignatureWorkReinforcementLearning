import os
import torch


def save_model(model, episode, optimizer, train_history, PATH, filename):
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_history,
    }, f"{PATH}/{filename}")


def load_model(save_path, algorithm, model_name, q, optimizer):
    # Get a list of all the saved models in the directory
    model_files = [f for f in os.listdir(save_path) if (f.endswith(".pt") and (model_name in f and algorithm in f))]

    if len(model_files) > 0:
        # Get the latest saved model (based on modification time)
        latest_model = max(model_files, key=lambda x: os.path.getmtime(f"{save_path}/" + x))

        checkpoint = torch.load(f"{save_path}/{latest_model}")

        # Load state from dictionary
        q.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_history = checkpoint['train_history']
        episode = checkpoint['episode']
        return q, optimizer, train_history, episode

    raise FileNotFoundError(f"No valid .pt file found in directory {save_path}")


def load_specific_model(model_path, q):
    checkpoint = torch.load(model_path)

    # Load state from dictionary
    q.load_state_dict(checkpoint['model_state_dict'])

    return q