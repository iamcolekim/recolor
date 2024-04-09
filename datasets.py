import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor
from skimage.color import rgb2lab, rgb2gray
from skimage.transform import resize

from PIL import Image


class GeneratorImageFolder(ImageFolder):

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img_original = self.transform(img)

            if not isinstance(img_original, np.ndarray):
                img_original = np.asarray(img_original)

            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255

            # Discard "L" layer
            img_ab = img_lab[:, :, 1:]
            # Resize to 56 x 56
            img_ab = resize(img_ab, (56, 56))
            # Convert to tensor, channel dimension first
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

            # Get L channel
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()  # Add additional dimension

        # Return L channel, ab channel
        # [1 x 224 x 224], [2 x 56 x 56]
        return img_original, img_ab
    


class RefinementImageFolder(ImageFolder):

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img_original = self.transform(img)

            # Apply resize on the PIL Image
            resize_56 = Resize((56, 56), interpolation=Image.BICUBIC)
            img_downsampled = resize_56(img_original)

            # Convert the PIL Images to tensors
            to_tensor = ToTensor()
            img_original = to_tensor(img_original)
            img_downsampled = to_tensor(img_downsampled)

        # Return full image, all three channels
        # [3 x 224 x 224]
        # apply also any other transformations
        return img_downsampled, img_original

class CombinedImageFolder(ImageFolder):

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img_original = self.transform(img)

            if not isinstance(img_original, np.ndarray):
                img_original = np.asarray(img_original)

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
