"""Implement a PCA to reduce the dimensionality of on image."""
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from skimage.metrics import structural_similarity
from tqdm import tqdm

np.random.seed(0)


def pca_transform(
    img_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform a PCA transformation on the input matrix.

    Args:
        img_rows (np.ndarray): Rows of flattened image data (size n x m).

    Returns:
        tuple: Instances of the Principal Component Analysis:
        np.ndarray (m x 1): The eigenvalues sorted in descending order.
        np.ndarray (m x m): The eigenvectors sorted according to the order of the eigenvalues.
        np.ndarray (n x m): The zero-meaned data matrix.
        np.ndarray (1 x m): The vector of the feature means of the input matrix.
    """
    # 4. compute mean for each feature
    # TODO
    # 5. center input data by subtracting mean
    # TODO
    # 6. calculate covariance matrix
    # TODO
    # 7. perform eigendecomposition of covariance matrix
    # TODO
    # 8. sort eigenvalues in descending order and eigenvectors by descending eigenvalues
    # TODO
    # 9. return sorted eigenvalues, eigenvectors, centered data and mean vector
    return None  # TODO


def pca_inverse_transform(
    data_centered: np.ndarray, eigenvec: np.ndarray, mean_v: np.ndarray, n_comp: int
) -> np.ndarray:
    """Perform PCA inverse transformation to reconstruct compressed data.

    Args:
        data_centered (np.ndarray): Data with zero mean.
        eigenvec (np.ndarray): The eigenvectors sorted according to the order of the PCA eigenvalues.
        mean_v (np.ndarray): The vector of the feature means.
        n_comp (int): The number of components for the reconstruction.

    Returns:
        np.ndarray: The reconstructed data.
    """
    # 10. select top 'n_comp' eigenvectors
    # TODO

    # 11. project centered data onto space defined by selected eigenvectors
    # TODO

    # 12. reconstruct reduced data
    # TODO
    # 13. return reconstructed data
    return None  # TODO


def expl_var(eigenval: np.ndarray, n_comp: int) -> float:
    """Calculate the cumulative explained variance ratio for a given number of components.

    Args:
        eigenval (np.ndarray): The eigenvalues sorted in descending order.
        n_comp (int): The number of components.

    Returns:
        float: Cumulative explained variance ratio.
    """
    # compute total variance
    total_variance = np.sum(eigenval)
    # compute cumulative explained variance
    cumulative_explained_var = np.sum(eigenval[:n_comp])
    # determine and return cumulative explained variance ratio
    cum_explained_variance_ratio = cumulative_explained_var / total_variance
    print(
        f"Explained Variance Ratio (n_comp={n_comp}): {cum_explained_variance_ratio:.2%}"
    )
    return cum_explained_variance_ratio


if __name__ == "__main__":
    # 1. create empty output dir
    # TODO

    # 2. load image, plot it and save as .png file
    # TODO

    # 3. reshape image in 2D-array of shape (num_rows, num_columns * num_channels)
    # TODO

    # 14. iterate through range of n_components values (possibly incrementing by 10)
    # TODO

    # 14.1. perform PCA using 'pca_transform' function
    # TODO
    # 14.2. reconstruct image from  lower-dimensional representation
    # using current number of components using 'pca_inverse_transform'
    # TODO
    # 14.3. reshape recovered image to its original shape
    # and save it in output folder
    # TODO

    # 14.4. compute and store cumulative explained variance ratio
    # for current number of components using 'expl_var' function
    # TODO

    # 14.5 compute and store SSIM
    # TODO

    # 15. plot cumulative explained variance ratios...
    # create list of x-axis tick locations (every 10th component)
    # TODO

    # 16. ...and SSIM
    # TODO
    pass
