import torch.utils.data as data
import torchvision.transforms as transforms

import datasets as ds
import models as colorization_models
from utils import *
import sys

if __name__ == "__main__":
    """
    This script trains the refinement model for a specified number of epochs.
    
    Usage: python train_refinement.py n_epochs mode
    
    Where n_epochs is the number of epochs and
    mode = 1 for lower augmentation and higher learning rate and mode = 2 for higher augmentation and lower learning
    rate
    """

    n_epochs = int(sys.argv[-1])
    mode = int(sys.argv[-2])

    if mode == 1:
        transforms_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.RandomResizedCrop(224, antialias=True),
            transforms.RandomHorizontalFlip()
        ])
    else:  # mode == 2
        transforms_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomResizedCrop(224, antialias=True),
            transforms.RandomHorizontalFlip()
        ])

    transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    dataset_train = ds.CombinedImageFolder("lhq_1024_jpg/train", transforms_train)
    dataset_val = ds.CombinedImageFolder("lhq_1024_jpg/val", transforms_val)

    batch_size = 64
    num_workers = 7

    dataloader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                       pin_memory=True)
    dataloader_val = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = colorization_models.RefinementModel()

    # Check previous checkpoints and load most recent epoch...
    checkpoint_files = [file for file in os.listdir("checkpoints") if
                        file.startswith("r_model_") and file.endswith(".pth")]

    start_epoch = 0

    if checkpoint_files:
        integers = [int(file.split('_')[-1].split('.')[0]) for file in checkpoint_files]
        largest_integer = max(integers)
        start_epoch = largest_integer + 1
        largest_checkpoint_file = f"checkpoints/r_model_{largest_integer}.pth"

        # Load existing weights
        print("Loading existing model weights")
        model.load_state_dict(torch.load(largest_checkpoint_file, map_location=torch.device("cpu")))

    model = model.to(device)

    # Create generator model and lock weights
    generator_model = colorization_models.GeneratorModel()
    generator_model.load_state_dict(torch.load("best_generator_model.pth", map_location=torch.device("cpu")),
                                    strict=False)
    generator_model = generator_model.to(device)

    # Fix weights
    for param in generator_model.parameters():
        param.requires_grad = False

    if mode == 1:
        learning_rate = 0.001
    else:  # mode == 2:
        learning_rate = 0.0002

    print("Training from epoch", start_epoch)
    train_net(model, dataloader_train, dataloader_val, learning_rate=learning_rate, num_epochs=n_epochs, mode="r",
              verbose=False, generator_model=generator_model, start_epoch=start_epoch)
