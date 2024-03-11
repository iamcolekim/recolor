#!/Users/colekim/anaconda3/envs/SchoolWork/bin/python
import torch
import numpy as np
import copy

# please be careful when importing this cleaner, as it has been adjusted to work more automatically 

#Path Handling
from pathlib import Path

# Dataset Handling
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Image Visualization
import matplotlib.pyplot as plt

from PIL import ImageCms



class RGBToLab:
    '''
    This class is used in the transforms.Compose function to convert images to the LAB color space as part of the data loading process
    '''
    def __call__(self, img):

        '''
        Step-by-step explanation of the code:

        The ImageCms.createProfile() function is a part of the ImageCms module in the Python Imaging Library (PIL), also known as Pillow. This function creates an ICC color profile.

        ImageCms.createProfile("sRGB"): This creates an ICC profile for the sRGB color space. An ICC profile is a set of data that characterizes a color input or output device, or a color space, according to standards promulgated by the International Color Consortium (ICC). The sRGB color space is a standard RGB color space created cooperatively by HP and Microsoft for use on monitors, printers, and the Internet.

        sRGB is the assumed default color space (over other RGB profiles like Adobe RGB, ProPhoto RGB, etc.) for pytorch and many other image processing libraries.

        ImageCms.createProfile("LAB"): This creates an ICC profile for the LAB color space. Assume that this conforms to ISO 15076-1:2010 and use PCSLAB as the profile connection space.

        ImageCms.buildTransformFromOpenProfiles(ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB")): This builds a transformation that can convert images from the sRGB color space to the LAB color space. It takes the ICC profiles for the sRGB and LAB color spaces as input.

        ImageCms.applyTransform(img, ImageCms.buildTransformFromOpenProfiles(ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"))): This applies the transformation to the input image. It takes the image and the transformation as input, and returns the image converted to the LAB color space.
        '''
        to_pil_image = transforms.ToPILImage()
        img = to_pil_image(img)
        sRGB_profile = ImageCms.createProfile("sRGB")
        LAB_profile = ImageCms.createProfile("LAB")

        transform = ImageCms.buildTransformFromOpenProfiles(sRGB_profile, LAB_profile, 'RGB', 'LAB')

        return ImageCms.applyTransform(img, transform)
    def __ref__(self):
        return "RGBToLab"
class NormalizeLAB:
    '''
    This class is used in the transforms.Compose function to convert images to the LAB color space as part of the data loading process
    '''
    # General formula for normalization: (x - mean) / std
    # LAB color space ranges: L [0,100], A [-128,127], B [-128,127]
    def __init__(self, mean = [50.0, -0.5, -0.5], std = [50.0, 127.5, 127.5]):
        self.mean = mean
        self.std = std

    def __call__(self, img):

        tensor = transforms.functional.to_tensor(img)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor
    def __ref__(self):
        return f"NormalizeLAB(mean={self.mean}, std={self.std} for L, A, B channels respectively)"

class SeparateChannels(object):
    def __call__(self, pic):
        L = pic[0, :, :].unsqueeze(0)
        AB = pic[1:, :, :]
        return {'L': L, 'AB': AB}
    def __ref__(self):
        return f"SeparateChannels"

transform = transforms.Compose([
    transforms.Resize(256), # Images resized to * x 256 or 256 x * (aspect ratio maintained)
    transforms.CenterCrop(224), # Images cropped to 224 x 224
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    RGBToLab(), # Convert images to the LAB color space
    NormalizeLAB(), # Normalize the LAB channels
])

def get_L_channel(x):
    return x['L']
def get_AB_channel(x):
    return x['AB']
transform_L = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    RGBToLab(),
    NormalizeLAB(),
    SeparateChannels(),
    get_L_channel
])

transform_AB = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    RGBToLab(),
    NormalizeLAB(),
    SeparateChannels(),
    get_AB_channel
])
transform_downsize = transforms.Compose([
    transforms.toPILImage(),
    transforms.Resize(28)
    transforms.toTensor()
])
transform_upsize = transforms.Compose([
    transforms.toPILImage(),
    transforms.Resize(28)
    transforms.toTensor()
])
dataset_dir = Path('./dataset/')

def main():
    #create the datasets
    dataset_all = ImageFolder(root=dataset_dir, transform=transform)
    dataset_L = ImageFolder(root=dataset_dir, transform=transform_L)
    dataset_AB = ImageFolder(root=dataset_dir, transform=transform_AB)
    
    dataset_AB_small = (copy.deepcopy(dataset_AB))
    dataset_AB_small.transform = transform_downsize

    dataset_L_small = (copy.deepcopy(dataset_L))
    dataset_L_small.transform = transform_downsize

    print(f"Finished Creating Datasets")

    #create the dataloaders
    dataloader_all = DataLoader(dataset_all, batch_size=64, shuffle=False, num_workers=4)
    dataloader_L = DataLoader(dataset_L, batch_size=64, shuffle=False, num_workers=4)
    dataloader_AB = DataLoader(dataset_AB, batch_size=64, shuffle=False, num_workers=4)
    dataloader_L_small = DataLoader(dataset_L_small, batch_size=64, shuffle=False, num_workers=4)
    dataloader_AB_small = DataLoader(dataset_AB_small, batch_size=64, shuffle=False, num_workers=4)
    print(f"Finished Creating Dataloaders")

    #print the shape of the dataset values
    for i, data in enumerate(dataloader_all):
        print(f"Batch {i} shape: {data[0].shape}")
        if i == 0:
            break
    for i, data in enumerate(dataloader_L):
        print(f"Batch {i} shape: {data[0].shape}")
        if i == 0:
            break
    for i, data in enumerate(dataloader_AB):
        print(f"Batch {i} shape: {data[0].shape}")
        if i == 0:
            break
    for i, data in enumerate(dataloader_L_small):
        print(f"Batch {i} shape: {data[0].shape}")
        if i == 0:
            break
    for i, data in enumerate(dataloader_AB_small):
        print(f"Batch {i} shape: {data[0].shape}")
        if i == 0:
            break
    
if __name__ == '__main__':
    main()