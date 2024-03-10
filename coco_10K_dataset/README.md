Use cleaner.py to find cleaned images.

/test includes images that were used to test certain cleaning functionality
though many tests were also performed on the coco_10K_dataset directly

threshold values were determined subjectively and using simple heuristics (e.g. tracking statistical deviations in color space, laplacians for blurring, minimum or maximums, frequencies for histogram analysis, etc.).
