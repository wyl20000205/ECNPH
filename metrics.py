import numpy as np
import torch
from config import cfg


def calc_loss(view1_predict, view2_predict, labels_1, labels_2):
    loss = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + (
        (view2_predict - labels_2.float()) ** 2
    ).sum(1).sqrt().mean()
    return loss


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    norm_x = torch.norm(x, dim=1, keepdim=True)
    norm_y = torch.norm(y, dim=1, keepdim=True)
    normalized_x = x / norm_x
    normalized_y = y / norm_y
    dist = torch.mm(normalized_x, normalized_y.t())
    return dist


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k_matrix(qB, rB, qL, rL, k=None):
    num_query = qL.shape[0]
    map = 0
    if k is None:
        k = rL.shape[0]
    for iter in range(num_query):
        gnd = (
            (qL[iter].unsqueeze(0).mm(rL.t()) > 0)
            .type(torch.float)
            .squeeze()
            .to(qB.device)
        )
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(qB.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex.to(cfg["device"]))
    map = map / num_query
    return map


def pr_curve(qB, rB, qL, rL):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (qL[i].unsqueeze(0).mm(rL.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[i, :], rB)
        tmp = (
            hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)
        ).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R


def p_top(qB, rB, qL, rL, K=[1, 200, 400, 600, 800, 1000]):
    num_query = qL.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (qL[iter].unsqueeze(0).mm(rL.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], rL.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


def compute_ndcg_at_n(qB, rB, qD, rD, n=1000):
    if torch.is_tensor(qB):
        qB = qB.cpu().numpy()
    if torch.is_tensor(rB):
        rB = rB.cpu().numpy()
    if torch.is_tensor(qD):
        qD = qD.cpu().numpy()
    if torch.is_tensor(rD):
        rD = rD.cpu().numpy()

    similarity = np.dot(qB, rB.T)
    retrieved_indices = np.argsort(-similarity, axis=1)[:, :n]

    ndcg_scores = []

    for i in range(qB.shape[0]):
        retrieved_labels = [rD[idx] for idx in retrieved_indices[i]]
        query_label = qD[i]

        # 改进的相关性计算
        rel = np.array(
            [
                (
                    len(set(query_label) & set(retrieved_labels[j]))
                    / len(set(query_label) | set(retrieved_labels[j]))
                    if len(set(query_label) | set(retrieved_labels[j])) > 0
                    else 0
                )
                for j in range(n)
            ]
        )

        dcg = np.sum(rel / np.log2(np.arange(2, n + 2)))
        ideal_rel = np.sort(rel)[::-1]
        idcg = np.sum(ideal_rel / np.log2(np.arange(2, n + 2)))

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(ndcg_scores)
