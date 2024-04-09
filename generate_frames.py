import warnings
warnings.filterwarnings("ignore")
import models as colorization_models
import datasets as ds

import torch.utils.data as data

from utils import *

def save_image(grayscale_input, ab_output, name):
    """
    Save predicted video frame as .png image.

    :param grayscale_input: The input tensor
    :param ab_output:       The output tensor
    :param name:            The name of the image
    """

    # Convert tensors to numpy
    grayscale_input = grayscale_input.detach().cpu().numpy().transpose((1, 2, 0))
    ab_output = ab_output.detach().cpu().numpy().transpose((1, 2, 0))

    color_image = np.concatenate([grayscale_input, ab_output], 2)

    # Unnormalize images
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128

    # Apply enhancement
    color_image_enhanced = color_image.copy()
    color_image_enhanced[:, :, 1:3] = np.clip(color_image_enhanced[:, :, 1:3] * 2, -128, 127)

    # Convert from LAB to RGB
    color_image = lab2rgb(color_image.astype(np.float64))
    color_image_enhanced = lab2rgb(color_image_enhanced.astype(np.float64))

    # Convert the 1-channel grayscale image to a 3-channel image
    grayscale_input = np.concatenate([grayscale_input, grayscale_input, grayscale_input], axis=-1)

    # Save images: grayscale images, colorized images, and 1.5x enhanced colorized images
    plt.imsave(arr=grayscale_input, fname=f"video_colorization/grayscale_frames/{name}", cmap="gray")
    plt.imsave(arr=color_image, fname=f"video_colorization/colorized_frames/{name}")
    plt.imsave(arr=color_image_enhanced, fname=f"video_colorization/colorized_frames_enhanced/{name}")

def generate_predictions(test_loader, net):
    """
    Generate video frame predictions.

    :param test_loader: The test dataloader
    :param net:         The combined ReColor model
    """
    print("Generating frames")

    net.eval()

    counter = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)

            # Forward pass
            outputs = net(inputs)

            # Save images
            for grayscale_input, ab_output in zip(inputs, outputs):
                save_image(grayscale_input, ab_output, name=f"{counter}.png")
                counter += 1

if __name__ == "__main__":
    """
    Script to generate colorized frames given black and white frames of a video.
    """

    dataset_test = ds.CombinedImageFolder("video_colorization/original_frames", None)

    batch_size = 32
    num_workers = 4

    dataloader_test = data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    refinement_model = colorization_models.RefinementModel()
    refinement_model.load_state_dict(torch.load("best_refinement_model.pth", map_location=torch.device("cpu")),
                                     strict=False)

    generator_model = colorization_models.GeneratorModel()
    generator_model.load_state_dict(torch.load("best_generator_model.pth", map_location=torch.device("cpu")),
                                    strict=False)

    model = colorization_models.RecolorModel(generator_model, refinement_model)
    model = model.to(device)

    generate_predictions(dataloader_test, model)
