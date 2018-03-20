import torch
import torch.nn.functional as F
from tqdm import tqdm


def recall_at_k(scores, ks=[1, 2, 3, 4, 5]):
    """
    Inputs:
    -------
    scores: matrix of (batch_size x n_response), where scores[0] is the true response.
    k: list of k values

    Outputs:
    --------
    recalls: list of recall@k / hits@k for k = 1...ks
    """
    _, sorted_idxs = torch.sort(scores, dim=1, descending=True)
    _, ranks = (sorted_idxs == 0).max(1)

    recalls = [((ranks + 1) <= k).float().mean() for k in ks]

    return recalls


def eval_model(model, data_iter, max_context_len, max_response_len, gpu=False):
    model.eval()
    scores = []

    valid_iter = tqdm(data_iter)
    valid_iter.set_description_str('Evaluation')

    for mb in valid_iter:
        context = mb.context[:, :max_context_len]

        # Get score for positive/ground-truth response
        score_pos = F.sigmoid(model(context, mb.positive[:, :max_response_len]).unsqueeze(1))
        # Get scores for negative samples
        score_negs = [
            F.sigmoid(model(context, getattr(mb, 'negative_{}'.format(i))[:, :max_response_len]).unsqueeze(1))
            for i in range(1, 10)
        ]
        # Total scores, positives at position zero
        scores_mb = torch.cat([score_pos, *score_negs], dim=1)
        print (scores_mb)
        scores.append(scores_mb)

    scores = torch.cat(scores, dim=0)
    recall_at_ks = [
        r.cpu().data[0] if gpu else r.data[0]
        for r in recall_at_k(scores)
    ]

    return recall_at_ks


def eval_pack_model(model, data_iter, max_context_len, max_response_len, gpu=False):
    model.eval()
    scores = []

    valid_iter = tqdm(data_iter)
    valid_iter.set_description_str('Evaluation')

    for mb in valid_iter:
        context = mb.context[0]
        pos = mb.positive[0]
        # print (context)
        cntx_l = mb.context[1]

        pos_l = mb.positive[1]
        neg_dat = [getattr(mb, 'negative_{}'.format(i))[0] for i in range(1,10) ]
        neg_lenghts = [getattr(mb, 'negative_{}'.format(i))[1] for i in range(1,10) ]
        #print (neg_dat)
        score_pos = F.sigmoid(model(context, cntx_l, pos, pos_l)).unsqueeze(1)
        # Get scores for negative samples
        score_negs = [
            F.sigmoid(model(context, cntx_l, neg_dat[i], neg_lenghts[i] ).unsqueeze(1))
            for i in range(0, 9)
        ]
        # Total scores, positives at position zero
        scores_mb = torch.cat([*score_negs, F.sigmoid], dim=1)
        #print (scores_mb)
        scores.append(scores_mb)

    scores = torch.cat(scores, dim=0)
    recall_at_ks = [
        r.cpu().data[0] if gpu else r.data[0]
        for r in recall_at_k(scores)
    ]

    return recall_at_ks


def eval_hybrid_model(model, dataset, gpu=False):
    model.eval()
    scores = []

    valid_iter = tqdm(dataset.valid_iter())
    valid_iter.set_description_str('Validation')

    for mb in valid_iter:
        # Get score for positive/ground-truth response
        score_pos = model(mb.context, mb.positive)[0].unsqueeze(1)
        # Get scores for negative samples
        score_negs = [
            model(mb.context, getattr(mb, 'negative_{}'.format(i)))[0].unsqueeze(1)
            for i in range(1, 10)
        ]
        # Total scores, positives at position zero
        scores_mb = torch.cat([score_pos, *score_negs], dim=1)

        scores.append(scores_mb)

    scores = torch.cat(scores, dim=0)
    recall_at_ks = [
        r.cpu().data[0] if gpu else r.data[0]
        for r in recall_at_k(scores)
    ]

    return recall_at_ks
