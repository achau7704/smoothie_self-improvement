import numpy as np
import torch
from torch.nn.functional import softmax

from src.utils import move_tensors_to_cpu, move_tensors_to_gpu

class Smoothie:
    def __init__(self, n_voters, dim):
        self.n_voters = n_voters
        self.dim = dim
        self.theta = np.ones(n_voters)

    def fit(self, lambda_arr: np.ndarray):
        """
        Fits weights using triplet method.

        Args:
            lambda (np.ndarray): embeddings from noisy voters. Has shape (n_samples, n_voters, dim)

        """
        n_samples, n_voters, dim = lambda_arr.shape

        diff = np.zeros(n_voters)  # E[||\lambda_i - y||^2]
        for i in range(n_voters):
            # Consider all other voters and select two at random
            other_idxs = np.delete(np.arange(n_voters), i)
            # Generate all unique pairs of indices
            rows, cols = np.triu_indices(len(other_idxs), k=1)
            pairs = np.vstack((other_idxs[rows], other_idxs[cols])).T

            index_diffs = []
            for j, k in pairs:
                index_diffs.append(
                    triplet(
                        lambda_arr[:, i, :], lambda_arr[:, j, :], lambda_arr[:, k, :]
                    )
                )

            # Set the difference to the average of all the differences
            diff[i] = np.mean(index_diffs)

        # Convert to cannonical parameters
        self.theta = dim / (2 * diff)
        self.theta = self.theta / self.theta.sum()

    def predict(self, lambda_arr: np.ndarray):
        """
        Predicts the true embedding using the weights

        Args:
            lambda_arr (np.ndarray): embeddings from noisy voters. Has shape (n_voters, dim)

        Returns:
            y_pred (np.ndarray): predicted true embedding. Has shape (dim)
        """
        predicted_y = 1 / self.theta.sum() * lambda_arr.T.dot(self.theta)
        return predicted_y


def triplet(i_arr: np.ndarray, j_arr: np.ndarray, k_arr: np.ndarray):
    """
    Applies triplet method to compute the difference between three voters

    Args:
        i_arr (np.ndarray): embeddings from voter i. Has shape (n_samples, dim)
        j_arr (np.ndarray): embeddings from voter j. Has shape (n_samples, dim)
        k_arr (np.ndarray): embeddings from voter k. Has shape (n_samples, dim)

    Returns:
        diff (float): difference between the three voters
    """
    diff_ij = (np.linalg.norm(i_arr - j_arr, axis=1, ord=2) ** 2).mean()
    diff_ik = (np.linalg.norm(i_arr - k_arr, axis=1, ord=2) ** 2).mean()
    diff_jk = (np.linalg.norm(j_arr - k_arr, axis=1, ord=2) ** 2).mean()
    return 0.5 * (diff_ij + diff_ik - diff_jk)
