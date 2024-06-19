import numpy as np
import torch
from torch.nn.functional import softmax

from src.utils import move_tensors_to_cpu, move_tensors_to_gpu


class Generator:
    """
    Corresponds to a LLM with a prompt prefix. The Generator is responsible for generating the next token given the current prompt.
    It handles moving KV cache between CPU and GPU and updating the token queue.
    """

    def __init__(self, model, tokenizer, prompt, max_length):
        """
        Args:
            model (torch.nn.Module): The LLM model
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer
            prompt (str): The prompt to use
            max_length (int): The maximum length of the model
        """

        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.prompt_token_ids = tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )
        self.device = self.model.device

        # Track the kv_cache
        self.kv_cache = None

        # The token queue stores the tokens that have been generated so far, but have not
        # yet been added to the prompt and are hence not included in the kv_cache.
        self.token_queue = []

    def update(self, token_id):
        """
        Updates the token_queue with the new token_id
        """
        self.token_queue.append(token_id)

    def construct_model_inputs(self):
        """
        Constructs the model inputs for the LLM.
        """

        if self.kv_cache is None:
            input_ids = self.prompt_token_ids
            attention_mask = torch.ones_like(input_ids)
        else:
            assert (
                len(self.token_queue) > 0
            ), "Generating input_ids without a token_queue"
            input_ids = torch.tensor(self.token_queue).view(1, -1)
            cur_seq_len = self.kv_cache[0][0].shape[2]
            attention_mask = torch.ones(1, input_ids.shape[1] + cur_seq_len)
        inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

        return inputs

    def get_W(self):
        """
        Returns the W token mapping matrix for the LLM
        """
        try:
            W = self.model.lm_head.weight.detach().data.numpy()
        except:
            W = self.model.embed_out.weight.detach().data.numpy()
        return W

    def get_output(self, return_probabilities=True) -> torch.Tensor:
        """
        Returns the probability vector over the next token.
        """
        inputs = self.construct_model_inputs()
        if self.kv_cache is None:
            outputs = self.model(**inputs)
        else:
            outputs = self.model(
                **inputs, past_key_values=move_tensors_to_gpu(self.kv_cache)
            )
        # Update the kv_cache and clear the token_queue
        self.kv_cache = move_tensors_to_cpu(outputs["past_key_values"])
        self.token_queue = []

        logits = outputs["logits"][:, -1, :].squeeze()

        if return_probabilities:
            # Compute probability vector
            probs = softmax(logits, dim=0)
            return probs.squeeze().detach().cpu().numpy()
        else:
            return logits.detach().cpu().numpy()


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
