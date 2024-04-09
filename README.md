# Recolor:  CNN-based Image Colorization Model
__Authors__: Jeff Chen, Cole Kim, Chloe Wu, Chloe Ha
___
Recolor is a multi-path CNN-based model that takes greyscale images and predicts colour channels. See [Architecture Section](#architecture) for more information.

### Getting Started
- [Jump to Dependencies Section and Install](#dependencies)
- Download Landscape Dataset. [Recommended -- Landscapes HQ Dataset](https://github.com/universome/alis/blob/master/lhq.md)
- [Review Model Training and Evaluation](#model-training-and-evaluation)
- [Run test scripts to generate model outputs](#dependencies)

### Architecture
As a multi-path architecture, the model leverages two separately trained neural network segments:
1) A Generator Network: used to predict coarse colour distribution predictions
2) A Refinement Network: uses high-resolution "hints" to fit and upscale the colour distributions to the full resolution

![High Level Architecture](./readme_auxfiles/high_level_model.png)

All networks utilize the CIELAB colour space (discussed as `Lab` henceforth).

The Generator model architecture takes in a 1x224x224 lightness input to obtain a 2x56x56 ab output. Batch normalization was considered for each layer before ReLU activation to aid in training performance and regularization. 

The refinement model takes a 1x224x224 lightness input and a low resolution 2x56x56 ab input, predicting the full 2x224x224 ab output. An encoder is used to extract features from the L input, before concatenating with separate encodings from the ab generator outputs. The combined encodings then pass through an EDSR-based super-resolution model.

Both models use transfer learning by leveraging low-level ResNet101 feature encodings (originally for classification).
![Detailed Architecture of Generator and Refinement Networks](./readme_auxfiles/architecture.png)


### Data Cleaning
Multiple outlier cleaning rules based on the workflow from Guo et al. (2023) were implemented.
Image outliers were grouped into 5 categories: corrupt, duplicate, near-monochromatic coloring,
skewed/unusual color distributions, and blurry. See datacleaner.py for more details
![Data Cleaning](./readme_auxfiles/data_cleaning.png)

### Dependencies

The following is a list of Python libraries required:
- <ins>General purpose</ins>: ```numpy```, ```pandas```
- <ins>Machine learning</ins>: ```torch```, ```torchvision```, ```scipy```, ```scikit-learn```
- <ins>Image processing</ins>: ```pillow```, ```cv2```, ```scikit-image```
- <ins>Visualizations and Metrics</ins>: ```matplotlib```, ```scikit-metrics```

