import numpy as np
from .utils import generate_batches, binary_log_loss

class MLPClassifier:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.001, epochs=200) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = 0.0001
    
    def fit(self, X, y):
        # TODO input data validations
        self.n_examples, self.n_features = X.shape
        self.n_outputs = y.shape[1]
        self.layer_units = [self.n_features] + list(self.hidden_layer_sizes) + [self.n_outputs]
        self.n_layers = len(self.layer_units)
        self.xdtype, self.ydtype = X.dtype, y.dtype
        self._initialize_weights()
        activations = [X]+[None]*(self.n_layers-1)
        deltas = [None]*(self.n_layers-1)
        coef_gradients = []
        intercept_gradients = []
        for i in range(self.n_layers-1):
            coef_gradients.append(np.empty(self.coefs[i].shape, dtype=self.xdtype))
            intercept_gradients.append(np.empty(self.intercepts[i].shape, dtype=self.xdtype))
        
        params = self.coefs + self.intercepts
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
                    batch_loss = self._backward_pass(X_batch, y_batch, activations, deltas, coef_gradients, intercept_gradients)
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
        except KeyboardInterrupt:
            print("Training interrupted by user.")

    def predict(self, X):
        y_pred = self._forward_pass(X, only_final_layer=True)
        y_pred = y_pred.ravel()
        y_pred[y_pred>=0.5] = 1
        y_pred[y_pred<0.5] = 0
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.average(y_pred == y)

    def _initialize_weights(self):
        self.iter = 0
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
        self.no_improvement_count = 0
        self.best_loss = np.inf
        self.validation_scores = None
        self.best_validation_score = None

    def _forward_pass(self, activations, only_final_layer=False):
        if only_final_layer:
            activation = activations
            for i in range(self.n_layers-1):
                activation = np.matmul(activation, self.coefs[i]) + self.intercepts[i]
                if i < self.n_layers-2:
                    activation = np.maximum(activation, 0, activation)
            if self.out_activation == "logistic":
                activation = 1/(1+np.exp(-activation))
            activations = activation
        else:
            for i in range(self.n_layers-1):
                activations[i+1] = np.matmul(activations[i], self.coefs[i]) + self.intercepts[i]
                if i < self.n_layers-2:
                    activations[i+1] = np.maximum(activations[i+1], 0, activations[i+1])
            if self.out_activation == "logistic":
                activations[i+1] = 1/(1+np.exp(-activations[i+1]))
        return activations

    def _backward_pass(self, X, y, activations, deltas, coef_gradients, intercept_gradients):
        loss = binary_log_loss(y, activations[-1])
        values = 0
        for c in self.coefs:
            c = c.ravel()
            values += np.dot(c, c)
        loss += 0.5 * self.alpha * values / self.n_examples
        last = self.n_layers - 2
        deltas[last] = activations[-1] - y
        coef_gradients[last] = np.matmul(activations[last].T, deltas[last]) + self.alpha * self.coefs[last]
        coef_gradients[last] /= self.n_examples
        intercept_gradients[last] = deltas[last].sum(axis=0) / self.n_examples
        for i in range(self.n_layers-2, 0, -1):
            deltas[i-1] = np.matmul(deltas[i], self.coefs[i].T)
            deltas[i-1][activations[i] <= 0] = 0
            coef_gradients[i-1] = np.matmul(activations[i-1].T, deltas[i-1]) + self.alpha * self.coefs[i-1]
            coef_gradients[i-1] /= self.n_examples
            intercept_gradients[i-1] = deltas[i-1].sum(axis=0) / self.n_examples
        return loss