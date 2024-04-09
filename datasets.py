import numpy as np
import torch
from torchvision.datasets import ImageFolder
from skimage.color import rgb2lab, rgb2gray
from skimage.transform import resize


class GeneratorImageFolder(ImageFolder):
    """
    ImageFolder used to create dataset for training generator model.
    """

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        img_original = img if self.transform is None else self.transform(img)

        # Convert image to np array if not already
        if not isinstance(img_original, np.ndarray):
            img_original = np.asarray(img_original)

        # Convert image to lab and normalize
        img_lab = rgb2lab(img_original)
        img_lab = (img_lab + 128) / 255

        # Discard "L" layer
        img_ab = img_lab[:, :, 1:]
        img_ab = resize(img_ab, (56, 56))

        # Convert to tensor, channel dimension first
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

        # Get L channel
        img_original = rgb2gray(img_original)
        img_original = torch.from_numpy(img_original).unsqueeze(0).float()  # Add additional dimension

        # Return L channel, ab channel
        # [1 x 224 x 224], [2 x 28 x 28]
        return img_original, img_ab


class CombinedImageFolder(ImageFolder):
    """
    ImageFolder used to create dataset for training refinement model.
    """

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        img_original = img if self.transform is None else self.transform(img)

        # Convert image to np array if not already
        if not isinstance(img_original, np.ndarray):
            img_original = np.asarray(img_original)

        # Convert image to lab and normalize
        img_lab = rgb2lab(img_original)
        img_lab = (img_lab + 128) / 255

        # Discard "L" layer
        img_ab = img_lab[:, :, 1:]

        # Convert to tensor, channel dimension first
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

        # Get L channel
        img_original = rgb2gray(img_original)
        img_original = torch.from_numpy(img_original).unsqueeze(0).float()  # Add additional dimension

        # Return L channel, ab channel
        # [1 x 224 x 224], [2 x 224 x 224]
        return img_original, img_ab
