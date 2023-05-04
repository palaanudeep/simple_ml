import numpy as np

def generate_batches(n_examples, batch_size):
    start = 0
    for i in range(int(n_examples/batch_size)):
        if start+batch_size <= n_examples:
            yield slice(start, start+batch_size)
            start += batch_size
    if start<n_examples:
        yield slice(start, n_examples)


def binary_log_loss(y_true, y_pred):
    return -((y_true*np.log(y_pred)).sum() + ((1-y_true)*np.log(1-y_pred)).sum())/y_pred.shape[0]

def log_loss(y_true, y_pred):
    # y_true = np.append(1 - y_true, y_true, axis=1)
    # y_pred = np.append(1 - y_pred, y_pred, axis=1)
    return -(y_true*np.log(y_pred)).sum()/y_pred.shape[0]

def get_y_type(y):
    # check if y is binary or multiclass
    if np.unique(y).shape[0] <= 2:
        return 'binary'
    else:
        return 'multiclass'
    