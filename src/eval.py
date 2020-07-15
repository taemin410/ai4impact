import torch


def calculate_rmse(ytrue, ypred):
    result = (ytrue, ypred) ** 2
    rmse = (torch.sum(result) / result.shape[0]) ** 0.5
    return rmse


def get_lagged_correlation(ypred, ytrue, num_delta):
    sig_pred = torch.std(ypred).float()
    vals = []

    for delta in range(num_delta):
        ytrue_d = ytrue[delta:]
        sig_true = torch.std(ytrue_d[delta:]).float()
        if delta == 0:
            val = (torch.mean(ypred * ytrue_d)) ** 0.5 / (sig_true * sig_pred)
        else:
            val = (torch.mean(ypred[:-delta] * ytrue_d)) ** 0.5 / (sig_true * sig_pred)
        vals.append(val)

    return vals
