import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from skimage.color import lab2rgb
from skimage.transform import resize

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(train_loader, net, criterion, optimizer, epoch):
    print(f"Training epoch: {epoch}")
    net = net.to(device)
    net.train()
    batch_time, data_time, train_losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    n_batches = len(train_loader)

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        data_time.update(time.time() - end)

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_losses.update(loss.item(), inputs.size(0))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time for forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print(
                f"Epoch {epoch} ({i + 1}/{n_batches}) | Time {batch_time.val:.3f} | Data {data_time.val:.3f} | Train Loss {train_losses.avg:.4f}")

    return train_losses.avg


def validate_epoch(val_loader, net, criterion, epoch):
    print(f"Validating epoch: {epoch}")

    net.eval()
    batch_time, data_time, val_losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    n_batches = len(val_loader)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            data_time.update(time.time() - end)

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_losses.update(loss.item(), inputs.size(0))

            # Record time for forward and backward passes
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                print(
                    f"Epoch {epoch} ({i + 1}/{n_batches}) | Time {batch_time.val:.3f} | Data {data_time.val:.3f} | Val Loss {val_losses.avg:.4f}")

    return val_losses.avg


def train_net(net, train_loader, val_loader, learning_rate=0.001, num_epochs=2, mode="g"):
    # Make checkpoints folder if does not previously exist
    os.makedirs("checkpoints", exist_ok=True)

    # Set manual seed for reproducability
    torch.manual_seed(1000)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Record training and validation losses for each epoch
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        train_loss[epoch] = train_epoch(train_loader, net, criterion, optimizer, epoch)
        val_loss[epoch] = validate_epoch(val_loader, net, criterion, epoch)

        # Step the learning rate scheduler
        exp_lr_scheduler.step()

        # Save the current model (checkpoint) to a file
        model_path = f"checkpoints/{mode}_model_{epoch + 1}"
        torch.save(net.state_dict(), model_path)
    print('Finished Training')

    # Write the train/test loss/err into CSV file for plotting later
    np.savetxt(f"{mode}_train_loss.csv", train_loss)
    np.savetxt(f"{mode}_val_loss.csv", val_loss)


def to_rgb(grayscale_input, ab_input, folder, name=None):
    plt.clf()

    grayscale_input = grayscale_input.detach().cpu().numpy().transpose((1, 2, 0))
    ab_input = ab_input.detach().cpu().numpy().transpose((1, 2, 0))

    if ab_input.shape[0] == 56 and ab_input.shape[1] == 56:
        # Upscale image
        ab_input = resize(ab_input, (224, 224), anti_aliasing=True)

    color_image = np.concatenate([grayscale_input, ab_input], 2)



    # Unnormalize
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128

    color_image_1, color_image_2, color_image_3 = color_image.copy(), color_image.copy(), color_image.copy()

    color_image_1[:, :, 1:3] = np.clip(color_image_1[:, :, 1:3] * 1, -128, 127)
    color_image_2[:, :, 1:3] = np.clip(color_image_2[:, :, 1:3] * 1.5, -128, 127)
    color_image_3[:, :, 1:3] = np.clip(color_image_3[:, :, 1:3] * 2, -128, 127)

    assert np.sum(color_image[:, :, 0:1] > 100) == 0
    assert np.sum(color_image[:, :, 0:1] < 0) == 0

    assert np.sum(color_image_1[:, :, 1:3] > 127) == 0
    assert np.sum(color_image_2[:, :, 1:3] > 127) == 0
    assert np.sum(color_image_3[:, :, 1:3] > 127) == 0

    assert np.sum(color_image_1[:, :, 1:3] < -128) == 0
    assert np.sum(color_image_2[:, :, 1:3] < -128) == 0
    assert np.sum(color_image_3[:, :, 1:3] < -128) == 0

    color_image_1 = lab2rgb(color_image_1.astype(np.float64))
    color_image_2 = lab2rgb(color_image_2.astype(np.float64))
    color_image_3 = lab2rgb(color_image_3.astype(np.float64))

    if name is not None:

        # Convert the 1-channel grayscale image to a 3-channel image
        grayscale_input = np.concatenate([grayscale_input, grayscale_input, grayscale_input], axis=-1)

        plt.imsave(arr=grayscale_input, fname='{}/{}'.format(folder["grayscale"], name), cmap="gray")
        plt.imsave(arr=color_image_1, fname='{}/{}'.format(folder["color1"], name))
        plt.imsave(arr=color_image_2, fname='{}/{}'.format(folder["color2"], name))
        plt.imsave(arr=color_image_3, fname='{}/{}'.format(folder["color3"], name))