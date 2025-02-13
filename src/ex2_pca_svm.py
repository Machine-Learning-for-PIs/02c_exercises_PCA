"""Use PCA to improve soft-margin SVM for face recognition."""

from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

def cv_svm(xtrain: np.ndarray, ytrain: np.ndarray) -> GridSearchCV:
    """Train and cross-validate a soft-margin SVM classifier with the grid search.

    Define an SVM classifier. Use the grid search and a 5-fold cross-validation
    to find the best value for the hyperparameters 'C' and 'kernel'.

    Args:
        xtrain (np.ndarray): The training data.
        ytrain (np.ndarray): The training labels.

    Returns:
        GridSearchCV: The trained model that was cross-validated with the grid search.
    """
    # define dictionary with parameter grids
    param_grid = {
        "C": [10**i for i in range(-2, 4, 1)],
        "kernel": ["rbf", "linear", "poly"],
    }
    # initialize svm classifier and perform grid search
    clf = GridSearchCV(svm.SVC(probability=True), param_grid)
    clf = clf.fit(xtrain, ytrain)
    # print parameters found with cross-validation
    print("Best estimator found with parameters:", clf.best_params_)
    return clf


def explained_var(xtrain: np.ndarray) -> np.ndarray:
    """Compute and plot the cumulative explained variance ratio for PCA.

    Calculate and plot the cumulative explained variance ratio for
    PCA applied to the training data.


    Args:
        xtrain (np.ndarray): The training data.

    Returns:
        np.ndarray: An array containing the cumulative explained variance ratios.
    """
    # 1. initialize PCA and fit on train data
    # TODO

    # 2. plot cumulative explained variance ratios
    # of each principal component against the number of components
    # TODO
    # 3. return array of cumulative explained variance ratios
    return None  # TODO


def pca_train(
    xtrain: np.ndarray, ytrain: np.ndarray, n_comp: int, train_fun: GridSearchCV
) -> tuple[PCA, GridSearchCV]:
    """Train a model with a given train function on the PCA-transformed data.

    Extract the top 'n_comp' principal components from the training data using PCA,
    and then trains a model on these components as features.

    Args:
        xtrain (np.ndarray): The training data.
        ytrain (np.ndarray): The training labels.
        n_comp (int): The number of PCA components.
        train_fun: The estimator function used to train the model.

    Returns:
        tuple: A tuple containing the PCA decomposition
               with 'n_comp' components and the model trained with the 'train_fun' function.
    """
    # 5. initialize PCA and fit on train data
    # TODO

    # 6. transform input data using PCA transform
    # TODO
    # 7. train model on transformed PCA features
    # TODO
    # 8. return PCA decomposition object and trained model
    return None  # TODO


if __name__ == "__main__":
    # load dataset 'Labeled Faces in the Wild' and get data,
    # take only classes with at least 70 images; downsize images for speed up
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    n_samples, h, w = lfw_people.images.shape
    x = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    # split data into training and test data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    # use 'StandardScaler' on train data and scale both train and test data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # 4. use `explained_var` function to calculate minimum number of components
    # needed to capture 90% of the variance; print this number
    # TODO

    # 9. use cv_svm function together with computed number of components
    # to call 'pca_train' and train model with reduced number of features,
    # measure time duration
    # TODO

    # 10. transform input test data using PCA transform
    # TODO
    # 11. compute and print accuracy of best model on test set
    # TODO

    # 12. use function 'cv_svm' to perform hyperparameter search with cross validation
    # on non-preprocessed train set, measure time
    # TODO

    # 13. compute and print accuracy of best model on test set
    # TODO

    # 14. (optional) plot top 12 eigenfaces
    # TODO

    pass
