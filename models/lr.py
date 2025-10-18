import numpy as np
from classes.kernels import get_kernel

def sigmoid(x):
    """
    Sigmoid function, common for both models. Method np.clip added in order
    to avoid both underflow and overflow.
    :param x: input value.
    :return: output of the sigmoid function.
    """
    z = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegression:
    """
        Logistic Regression class.
    """
    def __init__(self, learning_rate=0.001, n_iters=1000, random_state=None, lambda_reg=0.01):
        """
        Initialization of Logistic Regression model.
        :param learning_rate: coefficient for gradient descent steps;
        :param n_iters: number of iterations for training;
        :param random_state: random state used for reproducibility.
        :param lambda_reg: regularization term.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        self.random_state = random_state

    def calculate_loss(self, X, y):
        """
        Compute the loss of the prediction with respect to the true label.
        :param X: feature vector.
        :param y: true label of feature vector.
        :return: loss of the prediction.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        log_loss = np.mean(np.log2(1 + np.exp(-y * linear_model)))
        l2_regularization = (self.lambda_reg / 2) * np.dot(self.weights, self.weights)
        total_loss = log_loss + l2_regularization
        return total_loss

    def fit(self, X, y):
        """
        Method to perform training of the model.
        :param X: feature set.
        :param y: label set.
        :return: trained model.
        """
        # initialize weights and bias to zero
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        # set random seed if given
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # precompute log(2) for efficiency
        ln_2 = np.log(2)
        # train loop
        for t in range(self.n_iters):
            # extract a random indices
            idx = np.random.randint(0, n_samples)
            # take corresponding feature vector and label
            x_i = X[idx]
            y_i = y[idx]
            # prediction
            linear_model = np.dot(x_i, self.weights) + self.bias
            sigma_val = sigmoid(-y_i * linear_model)
            # compute gradient loss
            grad_loss_weights = -(sigma_val * y_i * x_i) / ln_2
            grad_loss_bias = -(sigma_val * y_i) / ln_2
            # regularization
            grad_reg = self.lambda_reg * self.weights
            total_grad_weights = grad_loss_weights + grad_reg
            # update
            self.weights -= self.lr * total_grad_weights
            self.bias -= self.lr * grad_loss_bias

            # print loss each 100 iterations
            if t % 100 == 0:
                loss = self.calculate_loss(X, y)
                self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        """
        Predict a probability for feature vector X.
        :param X: feature vector.
        :return: probability predicted.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predict label for feature vector X.
        :param X: feature vector.
        :param threshold: threshold for prediction.
        :return: label of feature vector. +1 if probability is >= 0.5, else -1.
        """
        return np.where(self.predict_proba(X) >= threshold, 1, -1)


class LogisticRegressionKernel:
    """
    Logistic Regression kernel class.
    """
    def __init__(self, kernel='gaussian', gamma=1.0, degree=2, normalize_kernel=True,
                 learning_rate=0.01, n_iters=1000, lambda_reg=0.01, random_state=None,
                 grad_clip=1e5, use_exact_reg=False):
        """
        Initialization of Logistic Regression model.
        :param kernel: type of kernel to employ.
        :param gamma: gamma for gaussian kernel.
        :param degree: degree for polynomial kernel.
        :param normalize_kernel: boolean for normalization on polynomial kernel.
        :param learning_rate: coefficient for gradient descent steps.
        :param n_iters: number of iterations on training.
        :param lambda_reg: regularization term.
        :param random_state: random state for reproducibility.
        :param grad_clip: coefficient to clip gradients.
        :param use_exact_reg: boolean to decide if using exact regularization or its approximation.
        """

        # parameters
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.grad_clip = grad_clip
        self.use_exact_reg = use_exact_reg

        self.alphas = None
        self.bias = 0.0
        self.X_train = None
        self.K_train = None
        self.loss_history = []

        kernel_params = {
            'gamma': gamma,
            'degree': degree,
            'normalize_kernel': normalize_kernel
        }
        self.kernel_func = get_kernel(kernel, **kernel_params)

    def calculate_loss(self, y):
        """
        Compute the loss of the predictions with respect to the true labels.
        :param y: true labels.
        :return: the loss occurred.
        """
        f_all = np.dot(self.K_train, self.alphas) + self.bias

        log_loss_terms = np.logaddexp(0, -y * f_all) / np.log(2)
        log_loss = np.mean(log_loss_terms)

        l2_regularization = (self.lambda_reg / 2) * np.dot(self.alphas.T, np.dot(self.K_train, self.alphas))
        return log_loss + l2_regularization

    def fit(self, X, y):
        """
        Perform training of the model.
        :param X: feature set.
        :param y: label set.
        :return: trained model.
        """
        # initialization of training
        # save training vectors
        self.X_train = X
        n_samples, _ = self.X_train.shape
        # initialize alpha and bias coefficients
        self.alphas = np.zeros(n_samples, dtype=np.float64)
        self.bias = 0.0

        # set random seed if indicated
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # compute kernel matrix
        self.K_train = self.kernel_func(self.X_train, self.X_train)
        # precompute log(2)
        ln_2 = np.log(2)

        for t in range(1, self.n_iters + 1):
            # adjust learning rate
            current_learning_rate = self.learning_rate / np.sqrt(t)
            # extract random index
            idx = np.random.randint(0, n_samples)
            # extract column
            k_i = self.K_train[idx, :]
            y_i = y[idx]

            # prediction
            linear_output = np.dot(k_i, self.alphas) + self.bias
            sigma_val = sigmoid(-y_i * linear_output)

            # loss of gradient for alphas and bias
            grad_loss_alphas = -(sigma_val * y_i * k_i) / ln_2
            grad_loss_bias = -(sigma_val * y_i) / ln_2

            # gradient norm
            # necessary to decide about the clip of the gradient (to avoid overflow)
            grad_norm = np.linalg.norm(grad_loss_alphas)
            if grad_norm > self.grad_clip:
                grad_loss_alphas = grad_loss_alphas * (self.grad_clip / grad_norm)

            # if True, apply exact regularization computation
            # while this ensure correctness of the process, computation time will be higher
            if self.use_exact_reg:
                grad_reg_alphas = self.lambda_reg * np.dot(self.K_train, self.alphas)
                total_grad_alphas = grad_loss_alphas + grad_reg_alphas
                self.alphas -= current_learning_rate * total_grad_alphas
                self.bias -= current_learning_rate * grad_loss_bias

            # else, apply weight decay technique
            else:
                self.alphas -= current_learning_rate * grad_loss_alphas
                decay_factor = current_learning_rate * self.lambda_reg
                self.alphas *= (1 - decay_factor)
                self.bias -= current_learning_rate * grad_loss_bias

            # print loss each 100 iterations
            if t % 100 == 0:
                current_loss = self.calculate_loss(y)
                self.loss_history.append(current_loss)

        return self

    def predict_proba(self, X):
        """
        Predict probability with respect to the label of the model.
        :param X: feature vector.
        :return: probability of class.
        """
        K_test = self.kernel_func(self.X_train, X)
        linear_output = np.dot(K_test.T, self.alphas) + self.bias
        return sigmoid(linear_output)

    def predict(self, X, threshold=0.5):
        """
        Predict class of a feature vector X.
        :param X: feature vector.
        :param threshold: threshold to classify: +1 if over, -1 otherwise.
        :return: predicted class.
        """
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= threshold, 1, -1)
