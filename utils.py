import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from skimage.color import lab2rgb
from skimage.transform import resize

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    """
    Utility class which stores running averages and most recent values during model training/testing/validation.
    """

    def __init__(self):
        # Initialize classes
        self.value = 0
        self.running_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the running average with a new value.

        :param val: The value to update the average with
        :param n:   Count to update the average with, defaults to 1
        """
        self.value = val
        self.sum += val * n
        self.count += n
        self.running_avg = self.sum / self.count


def train_epoch_g(train_loader, net, criterion, optimizer, epoch, verbose=False):
    """
    Train the generator model for one epoch.

    :param train_loader: The train dataloader
    :param net:          The generator model
    :param criterion:    The loss function to use
    :param optimizer:    The optimizer to use
    :param epoch:        The epoch number.
    :param verbose:      True to print additional loss information, False otherwise
    :return:             The average training loss for the training set
    """
    print(f"Training epoch: {epoch}")

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

        # Checkpoint model every few iterations
        if i % 20 == 0:
            model_path = f"checkpoints/recent_g.pth"
            torch.save(net.state_dict(), model_path)

        if verbose:
            print(
                f"Epoch {epoch} ({i + 1}/{n_batches}) | Time {batch_time.value:.3f} | Data {data_time.value:.3f} | Train Loss {train_losses.running_avg}")
        elif i % 20 == 0:
            print(
                f"Epoch {epoch} ({i + 1}/{n_batches}) | Time {batch_time.value:.3f} | Data {data_time.value:.3f} | Train Loss {train_losses.running_avg:.4f}")

    return train_losses.running_avg


def validate_epoch_g(val_loader, net, criterion, epoch, verbose=False):
    """
    Validate the generator model.

    :param val_loader: The train dataloader
    :param net:          The generator model
    :param criterion:    The loss function to use
    :param epoch:        The epoch number
    :param verbose:      True to print additional loss information, False otherwise
    :return              The average validation loss for the validation set
    """
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

            if verbose:
                print(
                    f"Epoch {epoch} ({i + 1}/{n_batches}) | Time {batch_time.value:.3f} | Data {data_time.value:.3f} | Val Loss {val_losses.running_avg}")
            elif i % 20 == 0:
                print(
                    f"Epoch {epoch} ({i + 1}/{n_batches}) | Time {batch_time.value:.3f} | Data {data_time.value:.3f} | Val Loss {val_losses.running_avg:.4f}")

    return val_losses.running_avg


def train_epoch_r(train_loader, net, criterion, optimizer, epoch, generator_model, verbose=False):
    """
   Train the refinement model for one epoch.

   :param train_loader:    The train dataloader
   :param net:             The refinement model
   :param criterion:       The loss function to use
   :param optimizer:       The optimizer to use
   :param epoch:           The epoch number
   :param generator_model: The generator model to use when getting blurry ab inputs
   :param verbose:         True to print additional loss information, False otherwise
   :return:                The average training loss for the training set
   """
    print(f"Training epoch: {epoch}")

    net.train()
    batch_time, data_time, train_losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    n_batches = len(train_loader)

    for i, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        data_time.update(time.time() - end)

        # Forward pass
        ab_blurry_pred = generator_model(inputs)
        outputs = net([inputs, ab_blurry_pred])
        loss = criterion(outputs, targets)
        train_losses.update(loss.item(), inputs[0].size(0))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time for forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Checkpoint model every few iterations
        if i % 20 == 0:
            model_path = f"checkpoints/recent_r.pth"
            torch.save(net.state_dict(), model_path)

        if verbose:
            print(
                f"Epoch {epoch} ({i + 1}/{n_batches}) | Time {batch_time.value:.3f} | Data {data_time.value:.3f} | Train Loss {train_losses.running_avg}")
        elif i % 20 == 0:
            print(
                f"Epoch {epoch} ({i + 1}/{n_batches}) | Time {batch_time.value:.3f} | Data {data_time.value:.3f} | Train Loss {train_losses.running_avg:.4f}")

    return train_losses.running_avg


def validate_epoch_r(val_loader, net, criterion, epoch, generator_model, verbose=False):
    """
   Validate the refinement model.

   :param val_loader:      The train dataloader
   :param net:             The refinement model
   :param criterion:       The loss function to use
   :param epoch:           The epoch number
   :param generator_model: The generator model to use when getting blurry ab inputs
   :param verbose:         True to print additional loss information, False otherwise
   :return                 The average validation loss for the validation set
   """
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
            ab_blurry_pred = generator_model(inputs)
            outputs = net([inputs, ab_blurry_pred])
            loss = criterion(outputs, targets)
            val_losses.update(loss.item(), inputs[0].size(0))

            # Record time for forward and backward passes
            batch_time.update(time.time() - end)
            end = time.time()

            if verbose:
                print(
                    f"Epoch {epoch} ({i + 1}/{n_batches}) | Time {batch_time.value:.3f} | Data {data_time.value:.3f} | Val Loss {val_losses.running_avg}")
            elif i % 20 == 0:
                print(
                    f"Epoch {epoch} ({i + 1}/{n_batches}) | Time {batch_time.value:.3f} | Data {data_time.value:.3f} | Val Loss {val_losses.running_avg:.4f}")

    return val_losses.running_avg


def train_net(net, train_loader, val_loader, learning_rate=0.001, num_epochs=1, mode="g", verbose=False,
              generator_model=None, start_epoch=0):
    """
    Code to train either the generator or the refinement model, with validation occuring at the end of every epoch.

    :param net:             The model to train
    :param train_loader:    The train dataloader
    :param val_loader:      The validation dataloader
    :param learning_rate:   The learning rate, for Adam optimizer
    :param num_epochs:      The number of epochs to train for
    :param mode:            "g" to train the generator model, "r" to train the refinement model
    :param verbose:         True to print additional loss information, False otherwise
    :param generator_model: The generator model to use, only for when training the refinement model
    :param start_epoch:     The epoch to start at
    """

    # Make checkpoints folder if does not previously exist
    os.makedirs("checkpoints", exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Record training and validation losses for each epoch
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for epoch in range(start_epoch, start_epoch + num_epochs):

        if mode == "g":
            train_loss[epoch - start_epoch] = train_epoch_g(train_loader, net, criterion, optimizer, epoch,
                                                            verbose=verbose)
            val_loss[epoch - start_epoch] = validate_epoch_g(val_loader, net, criterion, epoch, verbose=verbose)
        elif mode == "r":
            train_loss[epoch - start_epoch] = train_epoch_r(train_loader, net, criterion, optimizer, epoch,
                                                            generator_model, verbose=verbose)
            val_loss[epoch - start_epoch] = validate_epoch_r(val_loader, net, criterion, epoch, generator_model,
                                                             verbose=verbose)

        # Save the current model (checkpoint) to a file
        model_path = f"checkpoints/{mode}_model_{epoch}.pth"
        torch.save(net.state_dict(), model_path)

    print("Finished Training")

    # Write the train/test loss/err into CSV file for plotting later
    train_loss_location = f"{mode}_train_loss.csv"
    val_loss_location = f"{mode}_val_loss.csv"

    # Append train losses to csv to exists, create new csv otherwise
    if os.path.exists(train_loss_location):
        previous_train_loss = np.loadtxt(train_loss_location)
        np.savetxt(train_loss_location, np.concatenate([previous_train_loss.reshape(-1), train_loss]))
    else:
        np.savetxt(train_loss_location, train_loss)

    # Append validation losses to csv to exists, create new csv otherwise
    if os.path.exists(val_loss_location):
        previous_val_loss = np.loadtxt(val_loss_location)
        np.savetxt(val_loss_location, np.concatenate([previous_val_loss.reshape(-1), val_loss]))
    else:
        np.savetxt(val_loss_location, val_loss)


def to_rgb(grayscale_input, ab_input, ab_output, folder, name=None):
    """
    Convert model outputs to RGB and save as images.

    :param grayscale_input: The input tensor
    :param ab_input:        The output tensor
    :param ab_output:       The ground truth tensor
    :param folder:          The folder to save the image in
    :param name:            The name of the image
    """
    plt.clf()

    # Convert tensors to numpy
    grayscale_input = grayscale_input.detach().cpu().numpy().transpose((1, 2, 0))
    ab_input = ab_input.detach().cpu().numpy().transpose((1, 2, 0))
    ab_output = ab_output.detach().cpu().numpy().transpose((1, 2, 0))

    # Upscale output and ground truth if low resolution (for generator model outputs)
    if ab_input.shape[0] == 56 and ab_input.shape[1] == 56:
        # Upscale image
        ab_input = resize(ab_input, (224, 224), anti_aliasing=True)

    if ab_output.shape[0] == 56 and ab_output.shape[1] == 56:
        # Upscale image
        ab_output = resize(ab_output, (224, 224), anti_aliasing=True)

    color_image = np.concatenate([grayscale_input, ab_input], 2)
    ground_truth_color_image = np.concatenate([grayscale_input, ab_output], 2)

    # Unnormalize all images
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    ground_truth_color_image[:, :, 0:1] = ground_truth_color_image[:, :, 0:1] * 100
    ground_truth_color_image[:, :, 1:3] = ground_truth_color_image[:, :, 1:3] * 255 - 128
    ground_truth_color_image[:, :, 1:3] = np.clip(ground_truth_color_image[:, :, 1:3] * 1, -128, 127)

    # Create ab enhanced versions of the predicted images, clip to ensure stays within valid LAB range
    color_image_1, color_image_2, color_image_3 = color_image.copy(), color_image.copy(), color_image.copy()

    color_image_1[:, :, 1:3] = np.clip(color_image_1[:, :, 1:3] * 1, -128, 127)
    color_image_2[:, :, 1:3] = np.clip(color_image_2[:, :, 1:3] * 1.5, -128, 127)
    color_image_3[:, :, 1:3] = np.clip(color_image_3[:, :, 1:3] * 2, -128, 127)

    # Convert LAB images to RGB space
    color_image_1 = lab2rgb(color_image_1.astype(np.float64))
    color_image_2 = lab2rgb(color_image_2.astype(np.float64))
    color_image_3 = lab2rgb(color_image_3.astype(np.float64))
    ground_truth_color_image = lab2rgb(ground_truth_color_image.astype(np.float64))

    if name is not None:
        # Convert the 1-channel grayscale image to a 3-channel image
        grayscale_input = np.concatenate([grayscale_input, grayscale_input, grayscale_input], axis=-1)

        # Save all images:
        # 1. Grayscale image
        # 2. Predicted image, 1.0x ab scaling
        # 3. Predicted image, 1.5x ab scaling
        # 4. Predicted image, 2.0x ab scaling
        # 5. Ground truth image
        plt.imsave(arr=grayscale_input, fname='{}/{}'.format(folder["grayscale"], name), cmap="gray")
        plt.imsave(arr=color_image_1, fname='{}/{}'.format(folder["color1"], name))
        plt.imsave(arr=color_image_2, fname='{}/{}'.format(folder["color2"], name))
        plt.imsave(arr=color_image_3, fname='{}/{}'.format(folder["color3"], name))
        plt.imsave(arr=ground_truth_color_image, fname='{}/{}'.format(folder["ground_truth"], name))
