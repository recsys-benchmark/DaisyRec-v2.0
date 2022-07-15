import numpy as np
from daisy.utils.config import metrics_config

class Metric(object):
    def __init__(self, config) -> None:
        self.metrics = config['metrics']

    def run(self, test_ur, pred_ur, test_u):
        res = []
        for mc in self.metrics:
            kpi = metrics_config[mc](test_ur, pred_ur, test_u)
            res.append(kpi)
    
        return res

def Precision(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        pre = np.in1d(pred, list(gt)).sum() / len(pred)

        res.append(pre)

    return np.mean(res)

def Recall(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        rec = np.in1d(pred, list(gt)).sum() / len(gt)

        res.append(rec)

    return np.mean(res)

def MRR(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        for index, item in enumerate(pred):
            if item in gt:
                mrr = 1 / (index + 1)
                break
        
        res.append(mrr)

    return np.mean(res)

def MAP(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        r = np.in1d(pred, list(gt))
        out = [r[:k+1].sum() / (k + 1) for k in range(r.size) if r[k]]
        if not out:
            res.append(0.)
        else:
            ap = np.mean(out)
            res.append(ap)

    return np.mean(res)

def NDCG(test_ur, pred_ur, test_u):
    def DCG(r):
        r = np.asfarray(r) != 0
        if r.size:
            dcg = np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
            return dcg
        return 0.

    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        r = np.in1d(pred, list(gt))

        idcg = DCG(sorted(r, reverse=True))
        if not idcg:
            return 0.
        ndcg = DCG(r) / idcg

        res.append(ndcg)

    return np.mean(res)

def HR(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        res.append(1 if r.sum() else 0)

    return np.mean(res)

def AUC(test_ur, pred_ur, test_u):
    res = []

    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        pos_num = r.sum()
        neg_num = len(pred) - pos_num

        pos_rank_num = 0
        for j in range(len(r) - 1):
            if r[j]:
                pos_rank_num += np.sum(~r[j + 1:])

        auc = pos_rank_num / (pos_num * neg_num)
        res.append(auc)
                
    return np.mean(res)

def F1(test_ur, pred_ur, test_u):
    res = []

    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        pre = r.sum() / len(pred)
        rec = r.sum() / len(gt)

        f1 = 2 * pre * rec / (pre + rec)
        res.append(f1)

    return np.mean(res)
