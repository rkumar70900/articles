import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread

img = imread("./DSC_1624.JPG")   # <-- replace filename here
img_gray = rgb2gray(img)

U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)

k = int(0.1 * len(S)) 
img_reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :]

img_diff = np.abs(img_gray - img_reconstructed)

bytes_per_value = 8 
original_size_mb = (img_gray.size * bytes_per_value) / (1024 * 1024) 

compressed_size_values = U.shape[0] * k + k + k * Vt.shape[1] 
compressed_size_mb = (compressed_size_values * bytes_per_value) / (1024 * 1024) 

compression_ratio = (compressed_size_mb / original_size_mb) * 100 if original_size_mb > 0 else 0

print(f"\nOriginal image size: {original_size_mb:.2f} MB")
print(f"Compressed size (k={k}): {compressed_size_mb:.2f} MB")
print(f"Compression ratio: {compression_ratio:.2f}% of original size")
print(f"Data reduction: {100 - compression_ratio:.2f}%")
