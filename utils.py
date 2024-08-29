import math
import numpy as np

def cross_entropy(predicts, labels):
    """
    predicts: array of size(batch, category)
    labels: array of size(batch, )
    return: loss between predicts and labels
    """
    b, _ = predicts.shape
    matrix = np.zeros_like(predicts)
    idx_labels = np.vstack([np.arange(b), labels]).T
    matrix[idx_labels[:, 0], idx_labels[:, 1]] = 1
    
    return np.maximum(0, -np.sum(matrix * np.log(predicts + 1e-6), axis=1))

def label_to_one_hot(y, num_classes):
    if type(y) == int:
        y = [y]
    one_hot = np.zeros((len(y), num_classes), dtype=int)
    rows = np.arange(len(y))
    one_hot[rows, np.array(y)] = 1
    return one_hot


def mse_loss(y_hat, y):
    return np.sum(np.power(y_hat - y, 2))