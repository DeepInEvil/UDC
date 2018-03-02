def accuracy(y_pred, y_true, threshold=0.5, mean=True):
    y_pred = (y_pred >= threshold).float()
    tot = (y_pred == y_true).float()
    tot = tot.mean() if mean else tot.sum()
    return tot
