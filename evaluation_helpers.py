import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

import warnings

'''
Evaluation metrics are included in this file.
All evaluation assumes images are of the type np.ndarray
All evaluation functions return a float value
All evaluation functions take in two images as input of the form (height, width, channels)
'''

# use mean_squared_error(image1, image2) out of the box, returns float

def calculate_psnr(image_true, image_test, data_range=None):
    """
    Calculate the peak signal-to-noise ratio (PSNR) between two images.
    """

    # create data_range explicitly if not provided
    if data_range is None:
        assert(type(image_true) == np.ndarray and type(image_test) == np.ndarray), "images must be np.ndarrays"
        assert(image_true.shape == image_test.shape), "images must have the same dimensions"
        if (image_true.dtype != image_test.dtype):
            warnings.warn("image_true and image_test have different data types. "
                "Will default to the data type of image_true.")
            image_test = image_test.astype(image_true.dtype)
        
        # calculate data_range. assume data range does not exceed np type limits.
        data_range = np.ptp(image_true) # peak-to-peak value of the image
        
    
    return peak_signal_noise_ratio(image_true, image_test, data_range=data_range)


def calculate_ssim_channels(image_true, image_test, num_color_channels, data_ranges=None):
    """
    Calculate the structural similarity measure (SSIM) between two images.
    Performed on each channel of the image separately. 

    SSIM calculated using skimage.metrics.structural_similarity
    To match the implementation of Wang et al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, `use_sample_covariance` to False, and
    specify the `data_range` argument.
    """
    
    assert(type(image_true) == np.ndarray and type(image_test) == np.ndarray), "images must be np.ndarrays"
    assert(image_true.shape == image_test.shape), "images must have the same dimensions"
    

    MIN_DATA_RANGE = 1e-12 # minimum data range to avoid division by zero in some instances
    NUM_CHANNELS = image_true.shape[-1]
    
    ssim_values = []
    
    #force images to have higher precision
    image_true = image_true.astype(np.float64)
    image_test = image_test.astype(np.float64)


    for i in range(NUM_CHANNELS):
        
        # use provided data_range if available, otherwise calculate it
        if data_ranges is not None:
            data_range = data_ranges[i]
        else:
            # assume data range does not exceed np type limits.
            data_range = np.ptp(image_true[..., i]) # peak-to-peak value of the image
            data_range = data_range if data_range > MIN_DATA_RANGE else MIN_DATA_RANGE
            

        ssim = structural_similarity(image_true[..., i], image_test[..., i], data_range=data_range, gaussian_weights=True, use_sample_covariance=False, sigma=1.5)
        
        if np.isnan(ssim):
            print(f"Warning: NaN SSIM value encountered for channel {i}. Variance: {np.var(image_true[..., i])}, Data range: {data_range:.12f}")
        
        ssim_values.append(ssim)
    return tuple(ssim_values)

def calculate_ssim_avg(image_true, image_test, data_range = None):
    """
    Calculate the structural similarity measure (SSIM) between two images.
    Perform SSIM on each channel separately, then average the results. 
    Similar to old skimage implementation of SSIM for multichannel images.

    SSIM calculated using skimage.metrics.structural_similarity
    To match the implementation of Wang et al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, `use_sample_covariance` to False, and
    specify the `data_range` argument.
    """

    assert(type(image_true) == np.ndarray and type(image_test) == np.ndarray), "images must be np.ndarrays"
    assert(image_true.shape == image_test.shape), "images must have the same dimensions"

    MIN_DATA_RANGE = 1e-12 # minimum data range to avoid division by zero in some instances

    #force images to have higher precision
    image_true = image_true.astype(np.float64)
    image_test = image_test.astype(np.float64)
    
    # use provided data_range if available, otherwise calculate it
    if data_range is None:
        # assume data range does not exceed np type limits.
        data_range = np.ptp(image_true)  # peak-to-peak value of the image
        data_range = data_range if data_range > MIN_DATA_RANGE else MIN_DATA_RANGE
    
    NUM_CHANNELS = image_true.shape[-1]
    ssim_values = np.zeros(NUM_CHANNELS)
    for i in range(NUM_CHANNELS):
        ssim = structural_similarity(image_true[..., i], image_test[..., i], data_range=data_range, gaussian_weights=True, use_sample_covariance=False, sigma=1.5)
        
        if np.isnan(ssim):
            print(f"Warning: NaN SSIM value encountered for channel {i}. Variance: {np.var(image_true[..., i])}, Data range: {data_range:.12f}")

        ssim_values[i] = ssim
    return np.mean(ssim_values)