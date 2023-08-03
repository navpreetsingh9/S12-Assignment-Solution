import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from typing import Union
import torch
from torch import nn
from torchinfo import summary
import torchvision

def get_lr(optimizer):
    """
    Tracking learning rate during training
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_images(img_loader, classes):
    # get some random training images
    dataiter = iter(img_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:8]))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    
def print_summary(model, input_size=(1, 28, 28)):
    """Print Model summary

    Args:
        model (Net): Model Instance
        input_size (tuple, optional): Input size. Defaults to (1, 28, 28).
    """
    return summary(model, input_size=input_size)

def get_device() -> tuple:
    """Get Device type

    Returns:
        tuple: Device type
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return (use_cuda, device)

def load_weights_from_path(model, path):
    """load weights from file

    Args:
        model (Net): Model instance
        path (str): Path to weights file

    Returns:
        Net: loaded model
    """
    model.load_state_dict(torch.load(path))
    return model

def get_incorrrect_predictions(model, loader, device):
    """Get all incorrect predictions

    Args:
        model (Net): Trained model
        loader (DataLoader): instance of data loader
        device (str): Which device to use cuda/cpu

    Returns:
        list: list of all incorrect predictions and their corresponding details
    """
    model.eval()
    incorrect = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append(
                        [d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])

    return incorrect

def get_all_predictions(model, loader, device):
    """Get All predictions for model

    Args:
        model (Net): Trained Model 
        loader (Dataloader): instance of dataloader
        device (str): Which device to use cuda/cpu

    Returns:
        tuple: all predicted values and their targets
    """
    model.eval()
    all_preds = torch.tensor([]).to(device)
    all_targets = torch.tensor([]).to(device)
    with torch.no_grad():
        for data, target in loader:
            data, targets = data.to(device), target.to(device)
            all_targets = torch.cat(
                (all_targets, targets),
                dim=0
            )
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds = torch.cat(
                (all_preds, preds),
                dim=0
            )

    return all_preds, all_targets


def prepare_confusion_matrix(all_preds, all_targets, class_map):
    """Prepare Confusion matrix

    Args:
        all_preds (list): List of all predictions
        all_targets (list): List of all actule labels
        class_map (dict): Class names

    Returns:
        tensor: confusion matrix for size number of classes * number of classes
    """
    stacked = torch.stack((
        all_targets, all_preds
    ),
        dim=1
    ).type(torch.int64)

    no_classes = len(class_map)

    # Create temp confusion matrix
    confusion_matrix = torch.zeros(no_classes, no_classes, dtype=torch.int64)

    # Fill up confusion matrix with actual values
    for p in stacked:
        tl, pl = p.tolist()
        confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

    return confusion_matrix