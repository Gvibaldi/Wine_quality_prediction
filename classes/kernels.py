import numpy as np

def _calculate_sq_dists(X1, X2):
    """
    Function to compute the matrix of squared Euclidean distances.
    :param X1: feature vector X1;
    :param X2: feature vector X2.
    :return: squared distance between X1 and X2.
    """
    # ensure inputs are at least 2D
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sq_norms_X1 = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
    sq_norms_X2 = np.sum(X2 ** 2, axis=1)[np.newaxis, :]
    # squared distance computed using ||X1||^2 + ||X2||^2 - 2*(X1)^T*X2
    sq_dists = sq_norms_X1 + sq_norms_X2 - 2 * np.dot(X1, X2.T)
    # ensure non-negative distances due to potential floating point errors
    return np.maximum(sq_dists, 0)


def gaussian_kernel(X1, X2, gamma=1.0):
    """
    Computes the Gaussian (RBF) kernel between two sets of vectors.
        Formula: K(x, z) = exp(-||x - z||² / (2 * gamma))
    :param X1: feature vector X1;
    :param X2: feature vector X2;
    :param gamma.
    :return: kernel RBF between X1 and X2.
    """
    # obtain squared distance between X1 and X2
    sq_dists = _calculate_sq_dists(X1, X2)
    return np.exp(-sq_dists / (2 * gamma))


def polynomial_kernel(X1, X2, degree=3, normalize=False):
    """
    Computes the polynomial kernel between two sets of vectors.
    Handles optional normalization and scaling needed for Logistic Regression.
    Formula: K(x, z) = ( (x·z / d) + coef0 )^degree.
    :param X1: feature vector X1;
    :param X2: feature vector X2;
    :param degree: degree of polynomial;
    :param normalize: boolean value to apply normalization;
    :return: polynomial kernel between X1 and X2.
    """
    # ensure inputs are at least 2D
    x1_mat = np.atleast_2d(X1)
    x2_mat = np.atleast_2d(X2)

    # normalization
    if normalize:
        norm1 = np.linalg.norm(x1_mat, axis=1, keepdims=True)
        norm2 = np.linalg.norm(x2_mat, axis=1, keepdims=True)
        # avoid division by zero
        norm1[norm1 == 0] = 1.0
        norm2[norm2 == 0] = 1.0
        x1n = x1_mat / norm1
        x2n = x2_mat / norm2
    else:
        x1n = x1_mat
        x2n = x2_mat

    # dot product
    dot = np.dot(x1n, x2n.T)

    # compute Kernel Polynomial Function
    K = (dot + 1.0) ** degree

    # if the original inputs were 1D vectors, return a scalar value
    if X1.ndim == 1 and X2.ndim == 1:
        return K[0, 0]

    return K


def get_kernel(name, **params):
    """
    Function that returns a configured kernel function.
    :param name: name of the kernel function;
    :param params: parameters of kernel function.
    :return: configured kernel function.
    """
    # if kernel function is RBF
    if name == 'gaussian':
        # returns a function that only needs X1 and X2 as arguments
        return lambda X1, X2: gaussian_kernel(X1, X2, gamma=params.get('gamma', 1.0))

    # if kernel function is polynomial
    elif name == 'polynomial':
        return lambda X1, X2: polynomial_kernel(
            X1, X2,
            degree=params.get('degree', 3),
            normalize=params.get('normalize_kernel', False)
        )
    else:
        raise ValueError(f"Kernel '{name}' not supported")