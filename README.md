# Dimensionality Reduction Exercise

In this exercise, we will take a closer look at the mechanics of Principal Component Analysis (PCA). We will explore how PCA can reduce the complexity of our data and understand the practical benefits of this dimensionality reduction technique. Our first goal is to project our high-dimensional data onto a more compact feature space. We will then visualize how, even with this reduced set of features, we can retain the most information. This insight will serve as a basis for preprocessing the data that was used in our previous Support Vector Classification (SVC) exercise. We will observe the impact of this dimensionality reduction on our subsequent SVC training.

### Task 1: Principal Component Analysis

In this task, we will implement all the necessary steps to perform a PCA and visualize how much of the original information content of an image remains after the image features are projected into a lower dimensional space. 

Navigate to `src/ex1_pca.py` and have a look at the `__main__` function :

We create an empty directory called ``output`` using `os.makedirs`. Then we load the `statue.jpg` image from `data/images/` using ``imageio.imread``, plot it and save it into the ``output`` directory as ``original.png`` using `imsave` from `skimage.io`. 

Next, we reshape the image array into a 2D-array of shape $(d,n)$, where $d$ = `num_rows` is the number of features and $n$ = `num_columns * num_channels` would represent our examples.

Now we will implement the functions to perform a PCA transform and an inverse transform on our 2D array. First implement the function `pca_transform`: 

1. Compute the mean vector over the features of the input matrix. The resulting mean vector should have the size $(d,1)$. (Hint: use `keepdims=True`in the function `numpy.mean` to keep the dimensions for easier subtraction.)
2. Center the data by subtracting the mean from the 2D image array.
3. Compute the covariance matrix of the centered data. (Hint: `numpy.cov`.)
4. Perform the eigendecomposition of the covariance matrix. (Hint: `numpy.linalg.eigh`)
5. Sort eigenvalues in descending order and eigenvectors by their descending eigenvalues.
6. Return sorted eigenvalues, eigenvectors, centered data and the mean vector.

Next, implement the function `pca_inverse_transform`, which reconstructs the data using the top $n_comp$ principal components following these steps: 

7. Select the first $n_comp$ components from the given eigenvectors. 
8. Project the centered data onto the space defined by the selected eigenvectors by multiplying the transposed selected eigenvectors and the centered data matrix, giving us the reduced data.
9. Reconstruct the data projecting it back to the original space by multiplying the selected eigenvectors with the reduced data. Don't forget to add the mean vector afterwards.
10. Return the reconstructed data.

Go back to the `__main__` function. Now we perform PCA using the previously implemented `pca_transform` function.

We loop through a range of all possible values of the number of components. It is sufficient to use the step size of 10 to speed up the process. To monitor the progress of the loop, we create a progress bar using [the very handy Python package tqdm](https://github.com/tqdm/tqdm). Implement the following TODOs for the loop:


	11.1. Apply  the `pca_inverse_transform` function to project the image to lower-dimensional space using the current number of components and reconstruct the image from this reduced representation.

	11.2. Bring the resulting array back into the original image shape and save it in the ``output`` folder as an image called ``pca_k.png``, where _k_ is replaced with the number of components used to create the image.

	> Note: For `skimage.io` to be able to correctly interpret your image, you should cast it to the uint8 dtype, for example by using ``your_array.astype(np.uint8)``.
   
	11.3. Compute the cumulative explained variance ratio for the current number of components using the `expl_var` function and store it in a list for later plotting.

	11.4. We would also like to quantify how closely our created image resembles the original one. Use ``skimage.metrics.structural_similarity`` to compute a perceptual similarity score (SSIM) between the original and the reconstructed image and also store it in another list for later plotting. As we deal with RGB images, you have to pass `channel_axis=2` to the SSIM function.
   
We can now plot the cumulative explained variances of each principal component  and the corresponding SSIM values against the number of components. 

12. Look through the images you generated and find the one with the smallest _k_ which you would deem indistinguishable from the original. Compare this to both the explained variance and SSIM curves.
13. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.

### Task 2: PCA as Pre-processing (Optional)

We have seen that a significantly reduced feature dimensionality is often sufficient to effectively represent our data, especially in the case of image data. Building upon this insight, we will now revisit our Support Vector Classification (SVC) task, but this time with a preprocessing of our data using a PCA. Again, we will use the [Labeled Faces in the Wild Dataset](http://vis-www.cs.umass.edu/lfw/).

We start in the the `__main__` function.

We again load the dataset from ``sklearn.datasets.fetch_lfw_people`` in the same way as for the SVM Task and get access to the data. Then we split the data 80:20 into training and test data using `random_state=42` in the split function. 
And we apply the `StandardScaler` from `sklearn.preprocessing` on the train set and scale both the train and the test set.

Our goal now is to determine the minimum number of principal components needed to capture at least 90% of the variance in the data. First, implement the `explained_var` function:

1. Create an ``sklearn.decomposition.PCA`` instance and fit it to the data samples. Set `random_state=42` and `whiten=True` to normalize the components to have unit variance.
2. Plot the cumulative explained variance ratios of each principal component against the number of components using the ``explained_variance_ratio_`` property of the ``PCA`` instance. Note, that you have to sum up these ratios to get cumulative values (e.g. using ``np.cumsum``).
3. Return the array of cumulative explained variance ratios.

4. Return to the `__main__` function and use the `explained_var` function to calculate the minimum number of components needed to capture 90% of the variance. Print this number.

Implement the `pca_train` function to train a model on preprocessed data: 

5. Create a ``PCA`` instance and fit it to the data samples extracting the given number of components. Set `random_state=42` and `whiten=True`.
6. Project the input data on the orthonormal basis using the `PCA.transform`, resulting in a new dataset, where each sample is represented by the given number of the top principal components.
7. Call the `train_fun` function, which is passed as an argument, to train a model on the transformed PCA features.
8. The function should return a tuple containing two elements: the PCA decomposition object and the trained model.

9. Utilize our `cv_svm` function from the SVM exercise together with the computed number of required components to call `pca_train` in the `__main__` function.  This will allow us to train the model with the reduced feature set. Use the `time` function from the `time` module to measure and print the duration of this process for evaluation. 

10. To evaluate the model on the test set, we need to perform the same transform on the test data, as we did on the training data. Use the `PCA.transform` of your PCA decomposition object to do this.
11. Now we can compute and print the accuracy of our trained model on the test set.

12. In order to compare this model with the one without the PCA preprocessing, apply the function `cv_svm` on the original training set and measure the time.

13. Compute and print the accuracy of this trained model on the test set.
	
14. (Optional) You can use the `plot_image_matrix` function from `src/util_pca.py` to plot the top 12 eigenfaces.
	
15. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.
