import warnings
warnings.filterwarnings("ignore")
import models as colorization_models
import datasets as ds

import torch.utils.data as data
import torchvision.transforms as transforms

from utils import *

def test_net(test_loader, net, criterion, sel_folder = "test_outputs"):
    """
    Test the generator model and produce predictions as .png images.

    :param test_loader: The test dataloader
    :param net:         The trained generator model
    :param criterion:   The loss function
    :return:            The average test loss for the test set
    """
    print("Testing model")

    if sel_folder == "test_outputs":
        folders = {
            "grayscale": "test_outputs/generator/grayscale",
            "color1": "test_outputs/generator/colorized_1",
            "color2": "test_outputs/generator/colorized_1_5",
            "color3": "test_outputs/generator/colorized_2",
            "ground_truth": "test_outputs/generator/ground_truth"
        }
    elif sel_folder == "train_outputs":
        folders = {
            "grayscale": "train_outputs/generator/grayscale",
            "color1": "train_outputs/generator/colorized_1",
            "color2": "train_outputs/generator/colorized_1_5",
            "color3": "train_outputs/generator/colorized_2",
            "ground_truth": "train_outputs/generator/ground_truth"
        }
    elif sel_folder == "val_outputs":
        folders = {
            "grayscale": "val_outputs/generator/grayscale",
            "color1": "val_outputs/generator/colorized_1",
            "color2": "val_outputs/generator/colorized_1_5",
            "color3": "val_outputs/generator/colorized_2",
            "ground_truth": "val_outputs/generator/ground_truth"
        }

    # Make folders if they do not previously exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    net.eval()

    batch_time, data_time, test_losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    n_batches = len(test_loader)

    counter = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            data_time.update(time.time() - end)

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_losses.update(loss.item(), inputs.size(0))

            # Save images
            for grayscale_input, ab_input, ab_output in zip(inputs, outputs, targets):
                to_rgb(grayscale_input, ab_input, ab_output, folders, name=f"{counter}.png")
                counter += 1

            # Record time
            batch_time.update(time.time() - end)
            end = time.time()

            print(f"({i + 1}/{n_batches}) | Time {batch_time.value:.3f} | Data {data_time.value:.3f} | Test Loss {test_losses.running_avg:.4f}")

    return test_losses.running_avg


if __name__ == "__main__":
    """
    This script obtains the test loss for the generator model and outputs ground truth, lightness, and prediction
    images from the model (at various ab scaling factors).
    """

    # Val transforms = test transforms
    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    # Uncomment the line for the dataset path as necessary
    # dataset_path = "lhq_1024_jpg/test"
    # dataset_path = "lhq_1024_jpg/train"
    dataset_path = "lhq_1024_jpg/val"
    
    dataset_test = ds.GeneratorImageFolder(dataset_path, transforms_test)

    batch_size = 32
    num_workers = 4

    dataloader_test = data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = colorization_models.GeneratorModel()
    model.load_state_dict(torch.load("best_generator_model.pth", map_location=torch.device("cpu")), strict=False)

    model = model.to(device)

    criterion = nn.MSELoss()

    # provide sel_folder as needed to generate appropriate output folder
    # sel_folder = "test_outputs"
    # sel_folder = "train_outputs"
    sel_folder = "val_outputs"
    test_loss = test_net(dataloader_test, model, criterion, sel_folder = sel_folder)

    print("Test loss:", test_loss)
