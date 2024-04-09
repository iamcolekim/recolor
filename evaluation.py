import os
import cv2
import numpy as np
from evaluation_helpers import calculate_psnr, calculate_ssim_channels, calculate_ssim_avg

#Define the directories
#prefix = "test_outputs"
prefix = "train_outputs"
# prefix = "val_outputs"

true_images_dir = f"{prefix}/generator/ground_truth"
gen_images_dir = f"{prefix}/generator/colorized_1"
ref_images_dir = f"{prefix}/refinement/colorized_1"

# Get a sorted list of all files in both directories
true_images = sorted(os.listdir(true_images_dir))
gen_images = sorted(os.listdir(gen_images_dir))
ref_images = sorted(os.listdir(ref_images_dir))

def get_statistics(true_images_dir, gen_images_dir, ref_images_dir, true_images, gen_images, ref_images):
    # Initialize the statistics values
    psnr_values_gen = []
    psnr_values_ref = []
    ssim_values_channels_gen = [] #stores a tuple of ssim values for each channel
    ssim_values_channels_ref = [] #stores a tuple of ssim values for each channel
    ssim_avg_values_gen = []
    ssim_avg_values_ref = []

    # Iterate over the pairs of files
    pair = 0
    for true_file, gen_file, ref_file in zip(true_images, gen_images, ref_images):
        print("Processing pair", pair)
        pair += 1
        
        '''
        # for debugging:
        if pair == 10:
            break
        '''

        # Read the images. All images are in the same format HWC, where C = 3 typically
        true_image = cv2.imread(os.path.join(true_images_dir, true_file))
        gen_image = cv2.imread(os.path.join(gen_images_dir, gen_file))
        ref_image = cv2.imread(os.path.join(ref_images_dir, ref_file))

        # Calculate the PSNR
        psnr_gen = calculate_psnr(true_image, gen_image)
        psnr_values_gen.append(psnr_gen)
        psnr_ref = calculate_psnr(true_image, ref_image)
        psnr_values_ref.append(psnr_ref)


        # Calculate the SSIM per channel
        ssim_channels_gen = calculate_ssim_channels(true_image, gen_image, 3) #a tuple of ssim values per channel
        ssim_values_channels_gen.append(ssim_channels_gen) #this will be a list of tuples
        ssim_channels_ref = calculate_ssim_channels(true_image, ref_image, 3) #a tuple of ssim values per channel
        ssim_values_channels_ref.append(ssim_channels_ref) #this will be a list of tuples

        # Calculate the average SSIM
        ssim_avg_gen = calculate_ssim_avg(true_image, gen_image)
        ssim_avg_values_gen.append(ssim_avg_gen)
        ssim_avg_ref = calculate_ssim_avg(true_image, ref_image)
        ssim_avg_values_ref.append(ssim_avg_ref)
    
    # Calculate the mean values
    psnr_mean_gen = np.mean(psnr_values_gen)
    psnr_mean_ref = np.mean(psnr_values_ref)
    ssim_mean_gen = np.mean(ssim_avg_values_gen)
    ssim_mean_ref = np.mean(ssim_avg_values_ref)

    '''
    # for debugging:
    print(f"SSIM (Generator): {np.array(ssim_avg_values_gen)}")
    print(f"SSIM (Refinement): {np.array(ssim_avg_values_ref)}")
    '''


    return psnr_mean_gen, psnr_mean_ref, ssim_mean_gen, ssim_mean_ref

if __name__ == "__main__":
    psnr_mean_gen, psnr_mean_ref, ssim_mean_gen, ssim_mean_ref = get_statistics(true_images_dir, gen_images_dir, ref_images_dir, true_images, gen_images, ref_images)

    print("PSNR (Generator):", psnr_mean_gen)
    print("PSNR (Refinement):", psnr_mean_ref)
    print("SSIM (Generator):", ssim_mean_gen)
    print("SSIM (Refinement):", ssim_mean_ref)
    print("Accessing: ", prefix)