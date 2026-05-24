# utils/metrics.py

import tensorflow as tf
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

def score_function(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error = y_pred - y_true

    mask_early = error < 0  # early prediction
    mask_late = error >= 0  # late prediction

    score_early = tf.reduce_sum(tf.exp(-error[mask_early] / 13) - 1)
    score_late = tf.reduce_sum(tf.exp(error[mask_late] / 10) - 1)

    return score_early + score_late


def DAC(x_test, X_train, sigma, m=5):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(x_test)
    """
    Robust DAC calculation that avoids NaNs and invalid values
    """
    if len(X_train) < m:
        m = len(X_train)

    # Ensure input shapes are consistent
    assert x_test.shape[1] == X_train.shape[1], "Dimension mismatch"
    assert x_test.shape[0] == sigma.shape[0], "Length mismatch"

    tree = cKDTree(X_train)
    dists, _ = tree.query(x_test, k=m)

    # Compute average distance for each x_test point
    distances = np.mean(dists, axis=1) if m > 1 else dists

    # Remove NaNs and infs
    valid_mask = (~np.isnan(distances)) & (~np.isnan(sigma)) & (~np.isinf(distances)) & (~np.isinf(sigma))

    if np.sum(valid_mask) == 0:
        return np.nan, distances  # no valid data

    distances = distances[valid_mask]
    sigma = sigma[valid_mask]

    # Check for constant arrays or zero std deviation
    if np.std(distances) == 0 or np.std(sigma) == 0:
        return np.nan, distances

    # Check if arrays are effectively constant (within floating point tolerance)
    if np.allclose(distances, distances[0]) or np.allclose(sigma, sigma[0]):
        return np.nan, distances

    # Calculate Pearson correlation (DAC)
    DAC, _ = pearsonr(distances, sigma)
    return DAC, distances