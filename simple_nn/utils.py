import numpy as np

# generate batches from data
def generate_batches(n_examples, batch_size):
    start = 0
    for i in range(int(n_examples/batch_size)):
        if start+batch_size <= n_examples:
            yield slice(start, start+batch_size)
            start += batch_size
    if start<n_examples:
        yield slice(start, n_examples)

# binary cross-entropy loss
def binary_log_loss(y_true, y_pred):
    return -((y_true*np.log(y_pred)).sum() + ((1-y_true)*np.log(1-y_pred)).sum())/y_pred.shape[0]

# categorical cross-entropy loss
def log_loss(y_true, y_pred):
    return -(y_true*np.log(y_pred)).sum()/y_pred.shape[0]

def get_y_type(y):
    # check if y is binary or multiclass
    if np.unique(y).shape[0] <= 2:
        return 'binary'
    else:
        return 'multiclass'
    
def validate_input(X, y=None):
    # check if X is more than 2 dimensions or 0 and raise error
    if X.ndim > 2 or X.ndim == 0:
        raise ValueError('X should have 1 or 2 dimensions only')
    # check if y dtype is not int or float and raise error
    if isinstance(y, np.ndarray) and y.dtype not in [np.int64, np.float64]:
        raise ValueError('y should have int or float dtype')    
    # check if y is more than 2 dimensions and raise error
    if isinstance(y, np.ndarray) and y.ndim > 2:
        raise ValueError('y should have 1 or 2 dimensions only')
    # check if X and y have same number of examples and raise error
    if isinstance(y, np.ndarray) and X.shape[0] != y.shape[0]:
        raise ValueError('X and y should have same number of examples')
    # check if y is multi-label and raise error
    if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1:
        raise ValueError('Multi-label classification is not supported')
    # check if y contains same data type and raise error
    if isinstance(y, np.ndarray) and np.unique(y).shape[0] > 2 and np.unique(y).dtype != y.dtype:
        raise ValueError('y contains different data types')
    # check if X contains same data type and raise error
    if np.unique(X).dtype != X.dtype:
        raise ValueError('X contains different data types')
    return X, y
    