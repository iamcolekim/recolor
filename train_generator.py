import torch.utils.data as data
import torchvision.transforms as transforms

import datasets as ds
import models as colorization_models
from utils import *

if __name__ == "__main__":

    transforms_train = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        transforms.RandomResizedCrop(224, antialias=True),
        transforms.RandomHorizontalFlip()
    ])

    # Val transforms = test transforms
    transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    dataset_train = ds.GeneratorImageFolder("lhq_1024_jpg/train", transforms_train)
    dataset_val = ds.GeneratorImageFolder("lhq_1024_jpg/val", transforms_val)
    dataset_test = ds.GeneratorImageFolder("lhq_1024_jpg/test", transforms_val)

    batch_size = 128
    num_workers = 4

    dataloader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_val = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_test = data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = colorization_models.GeneratorModel()

    existing_checkpoints = "checkpoints/last_best.pth"

    if os.path.exists(existing_checkpoints):
        # Load existing weights
        print("Loading existing model weights")
        model.load_state_dict(torch.load(existing_checkpoints, map_location=torch.device("cpu")))

    model = model.to(device)

    train_net(model, dataloader_train, dataloader_val, learning_rate=0.001, num_epochs=14, mode="g")
