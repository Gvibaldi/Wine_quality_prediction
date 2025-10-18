import numpy as np
from classes.kernels import get_kernel

class SVM:
    """
    Class for linear Support Vector Machine (SVM).
    """
    def __init__(self, lambda_reg=0.01, n_iters=1000, random_state=None):
        """
        Initialization of SVM model.
        :param lambda_reg: lambda parameter for regularization on training;
        :param n_iters: number of iterations for training.
        :param random_state: random state for reproducibility.
        """
        # parameter for regularization
        self.lambda_reg = lambda_reg
        # number of iterations used for training
        self.n_iters = n_iters
        # weights
        self.w = None
        self.theta = None
        # random state
        self.random_state = random_state
        # loss history for SVM
        self.loss_history = []

    def calculate_loss(self, X, y, w):
        """
        Computation of hinge loss, defined as lambda/2 * ||w||^2 + mean(max(0, 1 - y * (Xw)))
        :param X: feature vector;
        :param y: label of feature vector;
        :param w: weights of model.
        :return:  value of hinge loss.
        """
        # mean(max(0, 1 - y * (Xw)))
        distances = 1 - y * np.dot(X, w)
        distances[distances < 0] = 0  # max(0, ...)
        hinge = np.mean(distances)
        # lambda/2 * ||w||^2
        reg = (self.lambda_reg / 2) * np.dot(w, w)
        # sum of them
        return reg + hinge

    def fit(self, X, y):
        """
        Training function of SVM model.
        :param X: feature set;
        :param y: label set.
        :return: trained model.
        """
        # transform [X] in [X, 1] to consider bias
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        # taking dimensions from X
        n_samples, n_features = X.shape
        # setting width of theta as the number of features for each example
        self.theta = np.zeros(n_features)
        # history of w
        w_history = []

        # setting random state for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # training cycle
        for t in range(1, self.n_iters + 1):
            # update of w
            w_current = (1 / (self.lambda_reg * t)) * self.theta
            # update of history of w
            w_history.append(w_current)
            # stochastic extraction of an element from training set
            idx = np.random.randint(0, n_samples)
            xi = X[idx]
            yi = y[idx]
            # update needed if conditions are not respected
            if yi * np.dot(w_current, xi) < 1:
                self.theta = self.theta + yi * xi

            if t % 100 == 0:
                loss = self.calculate_loss(X, y, w_current)
                self.loss_history.append(loss)

        # obtain weights
        all_w_matrix = np.stack(w_history, axis=0)
        self.w = np.mean(all_w_matrix, axis=0)
        return self

    # predictions or margins
    def predict(self, X, margins=False):
        """
        Compute predictions of SVM.
        :param X: feature set;
        :param margins: boolean to decide if return labels [-1, +1] or value of margins.
        :return: prediction.
        """
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        linear_output = np.dot(X, self.w)
        # if margins are required
        if margins:
            # return output
            return linear_output
        else:
            # return +1 or -1 by considering the sign
            return np.sign(linear_output)


class SVMKernel:
    """
    Class for linear Support Vector Machine (SVM) with kernel functions.
    """
    def __init__(self, kernel='gaussian', degree=3, gamma=1.0, lambda_reg=0.01, n_iters=1000, eta_b=0.01, random_state=None):
        """
        Initialization of SVM Kernel model.
        :param kernel: type of kernel function to apply;
        :param degree: degree of polynomial kernel function;
        :param gamma:  gamma for gaussian (RBF) kernel function;
        :param lambda_reg: lambda for regularization on training;
        :param n_iters: number of iterations for training;
        :param eta_b: learning rate for bias;
        :param random_state: random state used for reproducibility.
        """
        self.lambda_reg = lambda_reg
        self.n_iters = n_iters
        self.random_state = random_state
        self.alpha_bar = None
        self.X_train = None
        self.K_train = None
        self.bias = 0.0
        self.eta_b = eta_b
        self.loss_history = []

        # setting kernel parameters
        kernel_params = {
            'gamma': gamma,
            'degree': degree,
        }
        # setting kernel for SVM
        self.kernel_func = get_kernel(kernel, **kernel_params)
    def kernel(self, X1, X2):
        """
        Return kernel function.
        :param X1: feature vector X1;
        :param X2: feature vector X2.
        :return: kernel function applied to X1 and X2.
        """
        return self.kernel_func(X1, X2)

    # computation of loss (based on hinge loss for kernel functions)
    def calculate_loss(self, alpha, y, bias):
        reg_term = (self.lambda_reg / 2.0) * np.dot(np.dot(alpha.T, self.K_train), alpha)
        decision_values = np.dot(self.K_train, alpha) + bias
        hinge_loss = np.mean(np.maximum(0, 1 - y * decision_values))
        return reg_term + hinge_loss

    def fit(self, X, y):
        """
        Training function of SVM Kernel model.
        :param X: feature set;
        :param y: label set.
        :return: trained model.
        """
        # training set
        self.X_train = X
        # number of samples
        n_samples = len(y)
        # initialize alpha and beta with length = number of samples
        beta = np.zeros(n_samples)
        alpha_bar_sum = np.zeros(n_samples)
        # initialize bias
        bias, bias_sum = 0.0, 0.0

        # set random_state if indicated
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # compute kernel matrix
        self.K_train = self.kernel(self.X_train, self.X_train)

        # loop for number of iterations
        for t in range(1, self.n_iters + 1):
            # update alpha
            alpha_t = (1.0 / (self.lambda_reg * t)) * beta
            # random value for stochastic method
            idx = np.random.randint(0, n_samples)
            # extraction of kernel column for corresponding index
            K_i = self.K_train[:, idx]
            y_i = y[idx]
            # compute decision value
            decision_value_i = np.dot(alpha_t, K_i) + bias
            # adjustment of beta
            if y_i * decision_value_i < 1:
                beta[idx] += y_i
                bias += self.eta_b * y_i
            # alpha and bias updates
            alpha_bar_sum += (1.0 / (self.lambda_reg * t)) * beta
            bias_sum += bias
            # save loss each 100 iterations
            if t % 100 == 0:
                alpha_partial = alpha_bar_sum / t
                bias_partial = bias_sum / t
                current_loss = self.calculate_loss(alpha_partial, y, bias_partial)
                self.loss_history.append(current_loss)
        # set final alpha and bias
        self.alpha_bar = alpha_bar_sum / self.n_iters
        self.bias = bias_sum / self.n_iters
        return self

    # prediction function
    def predict(self, X_test, margins=False):
        """
        Compute predictions of SVM with kernel methods.
        :param X: feature set;
        :param margins: boolean to decide if return labels [-1, +1] or value of margins.
        :return: prediction.
        """
        if self.alpha_bar is None or self.X_train is None:
            raise RuntimeError("Model must be trained before predicting")
        K_test = self.kernel(self.X_train, X_test)
        decision_values = np.dot(self.alpha_bar, K_test) + self.bias
        if margins:
            return decision_values
        else:
            return np.sign(decision_values)