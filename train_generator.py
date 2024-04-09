import torch.utils.data as data
import torchvision.transforms as transforms

import datasets as ds
import models as colorization_models
from utils import *
import sys

if __name__ == "__main__":
    """
    This script trains the generator model for a specified number of epochs.

    Usage: python train_generator.py n_epochs

    Where n_epochs is the number of epochs
    """

    n_epochs = int(sys.argv[-1])

    transforms_train = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        transforms.RandomResizedCrop(224, antialias=True),
        transforms.RandomHorizontalFlip()
    ])

    transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    dataset_train = ds.GeneratorImageFolder("lhq_1024_jpg/train", transforms_train)
    dataset_val = ds.GeneratorImageFolder("lhq_1024_jpg/val", transforms_val)

    batch_size = 128
    num_workers = 7

    dataloader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_val = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = colorization_models.GeneratorModel()

    # Check previous checkpoints and load most recent epoch...
    checkpoint_files = [file for file in os.listdir("checkpoints") if
                        file.startswith("g_model_") and file.endswith(".pth")]

    start_epoch = 0

    if checkpoint_files:
        integers = [int(file.split('_')[-1].split('.')[0]) for file in checkpoint_files]
        largest_integer = max(integers)
        start_epoch = largest_integer + 1
        largest_checkpoint_file = f"checkpoints/g_model_{largest_integer}.pth"

        # Load existing weights
        print("Loading existing model weights")
        model.load_state_dict(torch.load(largest_checkpoint_file, map_location=torch.device("cpu")), strict=False)

    model = model.to(device)

    print("Training from epoch", start_epoch)
    train_net(model, dataloader_train, dataloader_val, learning_rate=0.001, num_epochs=n_epochs, mode="g",
              verbose=False, start_epoch=start_epoch)
