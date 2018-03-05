import torch


def recall_at_k(scores, ks=[1, 2, 3, 4, 5]):
    """
    Inputs:
    -------
    scores: matrix of (batch_size x n_response), where scores[0] is the true response.
    k: list of k values

    Outputs:
    --------
    recalls: recall@k / hits@k for k = 1...ks
    """
    _, sorted_idxs = torch.sort(scores, dim=1, descending=True)
    _, ranks = (sorted_idxs == 0).max(1)

    recalls = [((ranks + 1) <= k).float().mean() for k in ks]

    return recalls
