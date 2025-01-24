import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import os
from pathlib import Path

def gaussian_kernel(size, sigma):
    """
    Creates the Gaussian kernel.

    Args: 
        size (int): Size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: Gaussian kernel.
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def apply_gaussian_blur(image, kernel):
    """
    Apply Gaussian blur to an image using a kernel.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        kernel (np.ndarray): Gaussian kernel.

    Returns:
        np.ndarray: Blurred image.
    """
    if image.ndim == 3: 
        channels = [convolve2d(image[:, :, c], kernel, mode='same', boundary='symm') for c in range(3)]
        return np.stack(channels, axis=-1)
    else:  # For grayscale images
        return convolve2d(image, kernel, mode='same', boundary='symm')

def process_directory(input_dir, output_base_dir, kernel):
    """
    Apply Gaussian blur to all images in a directory.

    Args: 
        input_dir (str): Input directory path.
        output_base_dir (str): Output directory path (specific to dataset).
        kernel (np.ndarray): Gaussian kernel.
    """
    for root, _, files in os.walk(input_dir):
        # Get relative path and construct the output directory
        rel_path = os.path.relpath(root, input_dir)
        output_dir = os.path.join(output_base_dir, rel_path)

        # Create the output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for file in files:
            if file.startswith('.') or file.startswith('._'):
                continue
    
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_dir, file)

                try: 
                    image = Image.open(input_file).convert('RGB')
                    image_array = np.array(image)
                    blurred_array = apply_gaussian_blur(image_array, kernel)
                    blurred_image = Image.fromarray(np.clip(blurred_array, 0, 255).astype('uint8'))
                    blurred_image.save(output_file)
                except Exception as e:
                    print(f"Failed to process {input_file}: {e}")

# Parameters
kernel_size = 5
sigma = 1.0
gaussian_kernel_matrix = gaussian_kernel(kernel_size, sigma)

imagenet_dir = "data/ImageNet"
broden_dir = "data/broden1_224"
imagenet_output_dir = "data/ImageNet_blurred"
broden_output_dir = "data/broden1_224_blurred"

process_directory(imagenet_dir, imagenet_output_dir, gaussian_kernel_matrix)
process_directory(broden_dir, broden_output_dir, gaussian_kernel_matrix)