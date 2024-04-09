import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')
import models as colorization_models
import datasets as ds

import torch.utils.data as data
import torchvision.transforms as transforms

from utils import *

def test_net(test_loader, net, criterion):
    print("Testing model")

    folders = {
        "grayscale": "test/generator/grayscale",
        "color1": "test/generator/colorized_1",
        "color2": "test/generator/colorized_1_5",
        "color3": "test/generator/colorized_2"
    }

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
            for grayscale_input, ab_input in zip(inputs, outputs):
                to_rgb(grayscale_input, ab_input, folders, name=f"{counter}.png")
                counter += 1

            # Record time
            batch_time.update(time.time() - end)
            end = time.time()

            print(f"({i + 1}/{n_batches}) | Time {batch_time.val:.3f} | Data {data_time.val:.3f} | Test Loss {test_losses.avg:.4f}")

    return test_losses.avg


if __name__ == "__main__":

    # Val transforms = test transforms
    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    dataset_test = ds.RefinementImageFolder("lhq_1024_jpg/test", transforms_test)

    batch_size = 32
    num_workers = 3

    dataloader_test = data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = colorization_models.baseline_refine()
    #model.load_state_dict(torch.load("best_generator_model.pth", map_location=torch.device("cpu")))

    model = model.to(device)

    criterion = nn.MSELoss()
    test_loss = test_net(dataloader_test, model, criterion)

    print("Test loss:", test_loss)
