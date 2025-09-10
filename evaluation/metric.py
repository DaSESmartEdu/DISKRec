import torch
import time, sys
import torch.nn as nn


def gather_nd(params, indices):
    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1) 
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long() 
    m = 1
    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    out = torch.take(params, idx)
    return out.view(out_shape)


def auc(rate, negative, length):
    test = gather_nd(rate, negative)
    topk = torch.topk(test, 100).indices
    where = torch.where(topk == 99)
    auc = where[1]
    ran_auc = torch.randint(low=0, high=100, size=(length, 1), dtype=torch.int64).to(rate.device)
    auc = torch.mean(((auc - ran_auc) < 0).float())
    return auc.item()


def hr(rate, negative, length, k=5):
    test = gather_nd(rate, negative)
    topk = torch.topk(test, k).indices
    isIn = (topk == 99).float()
    row = torch.sum(isIn, dim=1)
    all_ = torch.sum(row)
    hr = all_ / length
    return hr.item()


def mrr(rate, negative, length):
    test = gather_nd(rate, negative)
    topk = torch.topk(test, 100).indices
    n = torch.where(topk == 99)[1]
    new_n = torch.add(n, 1)
    mrr_ = torch.sum(torch.reciprocal(new_n.float()))
    mrr = mrr_ / length
    return mrr.item()


def ndcg(rate, negative, length, k=5):
    test = gather_nd(rate, negative)
    topk = torch.topk(test, k).indices
    n = torch.where(topk == 99)[1]
    ndcg = torch.sum(torch.log2(torch.tensor(2.0).to(n.device)) / torch.log2(torch.add(n, 2).float())) / length
    return ndcg.item()


def env(rate, negative):
    length = negative.shape[0]
    hrat5 = hr(rate, negative, length, k=5)
    hrat10 = hr(rate, negative, length, k=10)
    ndcg5 = ndcg(rate, negative, length, k=5)
    ndcg10 = ndcg(rate, negative, length, k=10)
    mr = mrr(rate, negative, length)
    return hrat5, hrat10, ndcg5, ndcg10, mr

def BPR(pred_rating, negative):
    ratings = gather_nd(pred_rating, negative)
    pos_ratings = ratings[:, -1]
    neg_ratings = ratings[:, :-1]
    diff = pos_ratings.unsqueeze(1).repeat(1, neg_ratings.shape[1]) - neg_ratings
    loss = - nn.functional.logsigmoid(diff).mean()
    # loss = -diff.sigmoid().log().mean()
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total