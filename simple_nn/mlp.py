# importing libraries

import numpy as np
import matplotlib.pyplot as plt
from .utils import (
    generate_batches, 
    binary_log_loss, 
    log_loss, 
    get_y_type,
    validate_input
)

# MLPClassifier class
class MLPClassifier:

    # constructor
    # hidden_layer_sizes: tuple, default=(100,)
    # learning_rate: float, default=0.001
    # epochs: int, default=200
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.001, epochs=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = 0.0001
        self.isfitted = False
    
    # fit method
    # X: array-like of shape (n_examples, n_features)
    # y: array-like of shape (n_examples,)
    def fit(self, X, y):
        try:
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise Exception('X and y cannot be None')
            # validate input
            X, y = validate_input(X, y)
            # convert X dtype to float
            X = X.astype(np.float64, copy=False)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            print('Training...')
            self.y_type = get_y_type(y)
            self.classes = np.unique(y)
            self.n_examples, self.n_features = X.shape
            if self.y_type == "multiclass":
                # convert y to one-hot encoding
                y = np.eye(len(self.classes))[y.astype(int).ravel()]
            self.n_outputs = y.shape[1]
            self.layer_units = [self.n_features] + list(self.hidden_layer_sizes) + [self.n_outputs]
            self.n_layers = len(self.layer_units)
            self.xdtype, self.ydtype = X.dtype, y.dtype
            # initialize weights
            self._initialize_weights()
            # print('INFO', self.y_type, self.n_outputs, self.classes, self.out_activation)
            # setup activations and deltas for network layers
            activations = [X]+[None]*(self.n_layers-1)
            deltas = [None]*(self.n_layers-1)
            coef_gradients = []
            intercept_gradients = []
            for i in range(self.n_layers-1):
                coef_gradients.append(np.empty(self.coefs[i].shape, dtype=self.xdtype))
                intercept_gradients.append(np.empty(self.intercepts[i].shape, dtype=self.xdtype))
            
            params = self.coefs + self.intercepts
            # velocity initialization to control momentum of gradient descent
            self.velocities = [np.zeros_like(p) for p in params]
            batch_size = min(200, self.n_examples)
            try:
                for i in range(self.epochs):
                    total_loss = 0
                    for batch_slice in generate_batches(self.n_examples, batch_size):
                        X_batch = X[batch_slice]
                        y_batch = y[batch_slice]
                        activations[0] = X_batch
                        activations = self._forward_pass(activations)
                        batch_loss = self._backward_pass(y_batch, activations, deltas, coef_gradients, intercept_gradients)
                        # calculate total loss for each batch and update weights using momentum gradient descent optimizer
                        total_loss += batch_loss * (batch_slice.stop - batch_slice.start)
                        grads = coef_gradients + intercept_gradients
                        self.velocities = [0.9*velocity-self.learning_rate*grad for velocity, grad in zip(self.velocities, grads)]
                        updates = [0.9*velocity-self.learning_rate*grad for velocity, grad in zip(self.velocities, grads)]
                        for param, update in zip((p for p in params), updates):
                            param += update
                    self.iter += 1
                    self.loss = total_loss / self.n_examples
                    self.loss_curve.append(self.loss)
                    print(f'Epoch {i+1}/{self.epochs} loss: {self.loss:.4f}')
                self.isfitted = True
            except KeyboardInterrupt:
                print("Training interrupted by user.")
        except Exception as e:
            print(e)

    # predict method
    # X: array-like of shape (n_examples, n_features)
    def predict(self, X):
        try:
            # check if X is None
            if not isinstance(X, np.ndarray):
                raise Exception('X cannot be None')
            # check if model is fitted
            if not self.isfitted:
                raise Exception('Model not fitted yet. Call fit method first.')
            # validate input
            X, _ = validate_input(X)
            y_pred = self._forward_pass(X, only_final_layer=True)
            if self.n_outputs == 1:
                y_pred = y_pred.ravel()
            # convert 1-hot encoding to class labels
            if self.y_type == "multiclass":
                y_pred = y_pred.argmax(axis=1)
            else:
                y_pred[y_pred>=0.5] = 1
                y_pred[y_pred<0.5] = 0
            return self.classes[y_pred.astype(int)]
        except Exception as e:
            print(e)
        return None

    # score method
    # X: array-like of shape (n_examples, n_features)
    # y: array-like of shape (n_examples,)
    def score(self, X, y):
        try:
            # check if X is None or y is None
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise Exception('X and y cannot be None')
            # check if model is fitted
            if not self.isfitted:
                raise Exception('Model not fitted yet. Call fit method first.')
            # validate input and convert to 1-d array
            X, y = validate_input(X, y)
            y_pred = self.predict(X)
            y = y.ravel()
            return np.average(y_pred == y)
        except Exception as e:
            print(e)
        return None

    # initializing weights method
    def _initialize_weights(self):
        self.iter = 0
        if self.y_type == "multiclass":
            self.out_activation = "softmax"
        else:
            self.out_activation = "logistic"
        self.coefs, self.intercepts = [], []
        # normalized xavier weight initialization
        for i in range(self.n_layers - 1):
            coef_shape = (self.layer_units[i], self.layer_units[i+1])
            bound = np.sqrt(6 / (self.layer_units[i] + self.layer_units[i+1]))
            coef = np.random.uniform(-bound, bound, coef_shape)
            intercept = np.random.uniform(-bound, bound, self.layer_units[i+1])
            coef = coef.astype(self.xdtype, copy=False)
            intercept = intercept.astype(self.xdtype, copy=False)
            self.coefs.append(coef)
            self.intercepts.append(intercept)
        self.loss_curve = []

    # forward pass method
    # activations: list of arrays of shape (n_examples, n_features)
    # only_final_layer: bool, default=False
    def _forward_pass(self, activations, only_final_layer=False):
        # to calculate output layer activation
        if only_final_layer:
            activation = activations
            for i in range(self.n_layers-1):
                activation = np.matmul(activation, self.coefs[i]) + self.intercepts[i]
                if i < self.n_layers-2:
                    activation = np.maximum(activation, 0, activation)
            if self.out_activation == "logistic":
                activation = 1/(1+np.exp(-activation))
            elif self.out_activation == 'softmax':
                activation = np.exp(activation)
                activation /= activation.sum(axis=1, keepdims=True)
            activations = activation
        # to calculate activations for all layers
        else:
            for i in range(self.n_layers-1):
                activations[i+1] = np.matmul(activations[i], self.coefs[i]) + self.intercepts[i]
                if i < self.n_layers-2:
                    activations[i+1] = np.maximum(activations[i+1], 0, activations[i+1])
            if self.out_activation == "logistic":
                activations[i+1] = 1/(1+np.exp(-activations[i+1]))
            elif self.out_activation == 'softmax':
                activations[i+1] = np.exp(activations[i+1])
                activations[i+1] /= activations[i+1].sum(axis=1, keepdims=True)
        return activations

    # backward pass method
    # y: array-like of shape (n_examples,)
    # activations: list of arrays of shape (n_examples, n_features)
    # deltas: list of arrays of shape (n_examples, n_features)
    # coef_gradients: list of arrays of shape (n_examples, n_features)
    # intercept_gradients: list of arrays of shape (n_examples, n_features)
    def _backward_pass(self, y, activations, deltas, coef_gradients, intercept_gradients):
        # calculate loss
        if self.out_activation == "logistic":
            loss = binary_log_loss(y, activations[-1])
        else:
            loss = log_loss(y, activations[-1])
        values = 0
        for c in self.coefs:
            c = c.ravel()
            values += np.dot(c, c)
        loss += 0.5 * self.alpha * values / self.n_examples
        # calculate gradients for output layer
        last = self.n_layers - 2
        deltas[last] = activations[-1] - y
        coef_gradients[last] = np.matmul(activations[last].T, deltas[last]) + self.alpha * self.coefs[last]
        coef_gradients[last] /= self.n_examples
        intercept_gradients[last] = deltas[last].sum(axis=0) / self.n_examples
        # calculate gradients for hidden layers
        for i in range(self.n_layers-2, 0, -1):
            deltas[i-1] = np.matmul(deltas[i], self.coefs[i].T)
            deltas[i-1][activations[i] == 0] = 0
            coef_gradients[i-1] = np.matmul(activations[i-1].T, deltas[i-1]) + self.alpha * self.coefs[i-1]
            coef_gradients[i-1] /= self.n_examples
            intercept_gradients[i-1] = deltas[i-1].sum(axis=0) / self.n_examples
        return loss
    
    # plotting loss curve method
    def plot_loss_curve(self):
        try:
            plt.plot(self.loss_curve)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.show()
        except Exception as e:
            print(e)