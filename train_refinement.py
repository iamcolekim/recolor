import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import datasets as ds
import models as colorization_models
import matplotlib.pyplot as plt
from utils import *

if __name__ == "__main__":

    # transforms_train will have augmentation for the refinement model
    transforms_train = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        transforms.RandomResizedCrop(224, antialias=True),
        transforms.RandomHorizontalFlip(),
    ])

    # Val transforms = test transforms
    transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    dataset_train = ds.RefinementImageFolder("lhq_1024_jpg/train", transforms_train)
    dataset_val = ds.RefinementImageFolder("lhq_1024_jpg/val", transforms_val)
    dataset_test = ds.RefinementImageFolder("lhq_1024_jpg/test", transforms_val)

    batch_size = 32
    num_workers = 4

    dataloader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_val = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_test = data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    images, labels = next(iter(dataloader_train))
    image = images[0]
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # device = torch.device("mps") # MPS is not supported for BICUBIC upsample
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    #1.3*10^-5 is the baseline model loss

    model = colorization_models.RefinementModel(in_channels=3, rdb_channel_in=64, rdb_channel_out=64, k_size=3, scale_factor=4, rdb_depth=[2, 2])
    
    #some research to show that refinement model is better using RGB
    #model = colorization_models.baseline_refine()


    existing_checkpoints = "checkpoints/last_best.pth"

    if os.path.exists(existing_checkpoints):
        # Load existing weights
        print("Loading existing model weights")
        model.load_state_dict(torch.load(existing_checkpoints, map_location=torch.device("cpu")))

    model = model.to(device)

    train_net(model, dataloader_train, dataloader_val, learning_rate=0.0001, num_epochs=14, mode="g")
