#!/Users/colekim/anaconda3/envs/SchoolWork/bin/python
import torch
import numpy as np


#Path Handling
from pathlib import Path

# Dataset Handling
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Image Visualization
import matplotlib.pyplot as plt

# Image Processing
import cv2
from PIL import ImageCms
from PIL import Image
import hashlib

# Image Display
def display_image(filename):
    img = plt.imread(str(filename))
    plt.imshow(img)
    plt.show()


# Handling Outliers
def remove_images(image_files, images_to_remove, cleaning = False):
    # Create a set for faster lookup
    images_to_remove_set = set(images_to_remove)

    # Filter out the images to remove
    valid_images = [img_path for img_path in image_files if img_path not in images_to_remove_set]

    # If cleaning, delete the images from the file system
    if cleaning:
        for img_path in images_to_remove:
            Path(img_path).unlink()
    return valid_images

def ask_cleaning_type(type):
    # Function to ask for user confirmation before cleaning
    should_clean = input(f"Do you want to clean {type} images? (y/n): ")
    if should_clean.lower() == 'y':
        return True
    else:
        print(f"No cleaning will be done for {type} images") 
        return False
    
def ask_should_delete(filename, cleaning = False):
    # Function to ask for user confirmation before deleting
    if not cleaning:
        print(f"Detected, but not deleting {filename}")
        return True
    should_delete = input(f"Do you want to delete {filename}? (y/n): ")
    if should_delete.lower() == 'y':
        return True
    else:
        print(f"Skipping {filename}") 
        return False

def remove_corrupted(image_files):
    corrputed_images = []
    cleaning = ask_cleaning_type("corrupted")
    def is_corrupted_image(img_path):
        try:
            img = Image.open(img_path)
            img.verify()
            return False
        except (IOError, SyntaxError):
            return True
    for filename in image_files:
        if is_corrupted_image(filename):
            # Corrupted Images require no user confirmation
            #filename.unlink()
            print(f"Corrupted image: {str(filename)}")
            display_image(filename)
            if ask_should_delete(filename, cleaning):
                corrputed_images.append(filename)
    return remove_images(image_files, corrputed_images, cleaning)

def remove_duplicate_images(image_files):
    duplicate_images = []
    cleaning = ask_cleaning_type("duplicate")
    seen_hashes = set()
    for filename in image_files:
        with open(filename, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash not in seen_hashes:
            seen_hashes.add(file_hash)
        else:
            # Duplicates require no user confirmation
            #filename.unlink()
            print(f"Duplicate image: {str(filename)}")
            display_image(filename)
            if ask_should_delete(filename, cleaning):
                duplicate_images.append(filename)
    return remove_images(image_files, duplicate_images, cleaning)


def remove_extreme_aspect_ratio(image_files, min_ratio=0.5, max_ratio=2):
    meta_data_missing_images = []
    cleaning = ask_cleaning_type("extreme aspect ratio")
    def has_extreme_aspect_ratio(img_path, min_ratio, max_ratio):
        img = Image.open(img_path)
        width, height = img.size
        ratio = width / height
        return ratio < min_ratio or ratio > max_ratio
    for filename in image_files:
        if has_extreme_aspect_ratio(filename, min_ratio, max_ratio):
            print(f"Extreme aspect ratio: {str(filename)}")
            display_image(filename)
            if ask_should_delete(filename, cleaning):
                meta_data_missing_images.append(filename)
    return remove_images(image_files, meta_data_missing_images, cleaning)

def remove_monochrome_images(image_files):
    monochrome_images = []
    cleaning = ask_cleaning_type("monochrome")
    def is_monochrome(img_path, threshold=5):
        img = cv2.imread(img_path)
        if len(img.shape) < 3:
            return True
        if img.shape[2] == 1:
            return True
        std_dev = np.std(img, axis=(0, 1))
        return np.all(std_dev < threshold)
    
    for filename in image_files:
        if is_monochrome(str(filename)):
            print(f"Monochrome image: {str(filename)}")
            display_image(filename)
            if ask_should_delete(filename, cleaning):
                monochrome_images.append(filename)
    return remove_images(image_files, monochrome_images, cleaning)

def remove_unusual_color_distribution(image_files):
    unusual_color_images = []
    cleaning = ask_cleaning_type("unusual color distribution")
    def has_unusual_colors(img_path, threshold=0.7):
        img = cv2.imread(img_path)
        hist = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        max_val = np.max(hist)
        return max_val > threshold * np.prod(img.shape[:2])
    for filename in image_files:
        if has_unusual_colors(str(filename)):
            print(f"Unusual color distribution: {str(filename)}")
            display_image(filename)
            if ask_should_delete(filename, cleaning):
                unusual_color_images.append(filename)
    return remove_images(image_files, unusual_color_images, cleaning)

# Augmentation Handling Methods
#TODO: Add augmentation handling methods later

# Loader Handling Methods
'''
In PyTorch, transformations are usually implemented as callable classes, which means they implement the __call__ method. This method is called when an instance of the class is used as a function. The __call__ method takes an image as input and returns the transformed image.
'''
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
        
        return ImageCms.applyTransform(img, ImageCms.buildTransformFromOpenProfiles(
            ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB")))

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

transforms = transforms.Compose([
    transforms.Resize(256), # Images resized to * x 256 or 256 x * (aspect ratio maintained)
    transforms.CenterCrop(224), # Images cropped to 224 x 224
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    RGBToLab(), # Convert images to the LAB color space
    NormalizeLAB(), # Normalize the LAB channels
])

# Get the list of images in the dataset
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp") #all supported image extensions of TorchVision

image_files = list(dataset_dir.glob('**/*'))
image_files = [img for img in image_files if img.suffix in IMG_EXTENSIONS]

# driver code
def main(image_files):

    # Get the list of images in the dataset
    

    # Display the first image in the dataset
    display_image(image_files[0])

    # Handle outliers in the dataset

    # Remove non-starters (easy detection, always delete)
    image_files = remove_corrupted(image_files)
    image_files = remove_duplicate_images(image_files)

    # Remove with user-confirmation (human verification necessary)
    image_files = remove_extreme_aspect_ratio(image_files)
    image_files = remove_monochrome_images(image_files)
    image_files = remove_unusual_color_distribution(image_files)

    # Load the image dataset folder as an ImageFolder object
    dataset_dir = Path('coco_10K_dataset/test/')
    dataset = ImageFolder(root=dataset_dir, transform=outlier_transform, is_valid_file=lambda x: Path(x) in image_files)

    # Prepare the dataset for loading
    dataset.transform = dataloading_transform

    # Create a dataloader from the dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    main(image_files)