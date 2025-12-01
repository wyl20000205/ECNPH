import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from scipy.linalg import hadamard
from scipy.optimize import linear_sum_assignment
from einops import rearrange
import torch.nn.functional as F


class KEY_LOSS(object):
    # "will release after the paper accepted"
    pass


class KL_LOSS(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = cfg["device"]
        self.tmp = 0.1

    def forward(self, x, y):
        p_image = F.softmax(x / self.tmp, dim=-1)
        p_text = F.softmax(y / self.tmp, dim=-1)

        kl_loss = F.kl_div(
            p_image.log(),  # 输入需要是log概率
            p_text,  # 目标概率分布
            reduction="batchmean",  # 对整个batch求平均
        )

        return kl_loss


class DPDH_LOSS(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = cfg["device"]
        torch.manual_seed(cfg["seed"])

        self.alpha_a = 0.99
        self.alpha_b = 0.1
        self.alpha_prompt_margin = 0.2

        self.cross_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.prompt_img_dict = nn.ParameterDict()
        self.prompt_txt_dict = nn.ParameterDict()

        for one in cfg["list_bit"]:
            key = str(one)
            param_img = nn.Parameter(torch.randn(cfg["num_class"], one).to(self.device))
            param_txt = nn.Parameter(torch.randn(cfg["num_class"], one).to(self.device))
            nn.init.xavier_uniform_(param_img)
            nn.init.xavier_uniform_(param_txt)
            self.prompt_img_dict[key] = param_img
            self.prompt_txt_dict[key] = param_txt

        self.prompt_scale = nn.Parameter(
            torch.tensor(self.alpha_prompt_margin).to(self.device)
        )

    def forward(self, x, y, label, img_pre, txt_pre):
        one = str(x.shape[1])

        d_loss = self.cross_loss(img_pre, torch.argmax(label, -1)) + self.cross_loss(
            txt_pre, torch.argmax(label, -1)
        )

        bce_loss = self.bce_loss(txt_pre, label) + self.bce_loss(img_pre, label)

        image_prompt = self.prompt_img_dict[one]
        text_prompt = self.prompt_txt_dict[one]

        image_prompt_response = (
            F.normalize(x, p=2, dim=1) @ F.normalize(image_prompt, p=2, dim=1).T
        )
        text_prompt_response = (
            F.normalize(y, p=2, dim=1) @ F.normalize(text_prompt, p=2, dim=1).T
        )

        pre_image = x + self.prompt_scale * (image_prompt_response @ image_prompt)
        pre_text = y + self.prompt_scale * (text_prompt_response @ text_prompt)

        dual_loss = self.alpha_a * self.prompt_driven_loss_a(
            pre_image, pre_text, label
        ) + self.alpha_b * self.prompt_driven_loss_b(
            image_prompt_response, text_prompt_response, label
        )

        return dual_loss + bce_loss + d_loss

    def prompt_driven_loss_a(self, image_embed, text_embed, labels):
        similarity_matrix = image_embed @ text_embed.T
        labels_sim = (labels @ labels.T) > 0
        labels_sim = labels_sim.float().to(self.device)
        positive_mask = labels_sim
        negative_mask = 1 - labels_sim
        positive_loss = (1.0 - similarity_matrix) * positive_mask
        negative_loss = F.softplus(similarity_matrix - 0.1) * negative_mask
        return (positive_loss.sum() + negative_loss.sum()) / (
            positive_mask.sum() + negative_mask.sum() + 1e-8
        )

    def prompt_driven_loss_b(self, image_hash, text_hash, labels, margin=0.1):
        similarity_matrix = (
            F.normalize(image_hash, p=2, dim=1) @ F.normalize(text_hash, p=2, dim=1).T
        )
        labels_sim = (labels @ labels.T > 0).float().to(self.device)
        positive_mask = labels_sim
        negative_mask = 1 - labels_sim
        positive_loss = (1.0 - similarity_matrix) * positive_mask
        negative_loss = (
            F.softplus(similarity_matrix.clamp(min=-10, max=10) - margin)
            * negative_mask
        )
        return (positive_loss.sum() + negative_loss.sum()) / (
            positive_mask.sum() + negative_mask.sum() + 1e-8
        )


class DNPH_LOSS(nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(
            (torch.randn(cfg["num_class"], cfg["num_bit"]) / 8).to(cfg["device"])
        )
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mrg = 1.0

    def forward(
        self, feature_1, feature_2, predict_1, predict_2, label_1, label_2
    ):  # origin

        d_loss = self.cross_entropy(
            predict_1, torch.argmax(label_1, -1)
        ) + self.cross_entropy(predict_2, torch.argmax(label_2, -1))

        loss = d_loss
        return loss

    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    # print(c.min(), c.mean())
                    break
        return hash_targets


class Cross_modal_class_balance_loss(nn.Module):
    def __init__(self, args, bit, gamma=2.0, alpha=0.25):
        super(Cross_modal_class_balance_loss, self).__init__()
        self.hash_targets = self.get_hash_targets(args.numclass, bit).to(args.device)
        self.args = args
        self.multi_label_random_center = (
            torch.randint(2, (bit,)).float().to(args.device)
        )
        self.balance_weight = torch.tensor(args.balance_weight).float().to(args.device)
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def reszet_parameters(self):
        # Initial weight
        self.weight.data.fill_(0.5)
        # self.weight.data.fill_(1)

    def hloss(self, outputs, targets):

        if torch.min(targets) >= 0:
            targets = 2 * targets - 1

        # 计算Hinge Loss
        hinge_loss = 1 - outputs * targets
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss.mean()

    def forward(self, u, y, args):  # hash_text label
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.hloss(u, hash_center)
        Q_loss = (u.abs() - 1).pow(3).mean()
        y = y.float().to(self.args.device)

        balance_loss = self.balance_weight * ((y.mean(dim=0) - 0.5).abs().mean())

        return torch.sigmoid(self.weight) * (
            center_loss + args.lambda1 * Q_loss + balance_loss
        )

    def label2center(self, y):
        # 将标签转成float 放在cpu上
        y = y.float().to(self.args.device)
        center_sum = y @ self.hash_targets
        random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
        center_sum[center_sum == 0] = random_center[center_sum == 0]
        hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()
        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for _ in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    # print(c.min(), c.mean())
                    break
        return hash_targets


class DSPH_LOSS(nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        torch.manual_seed(cfg["seed"])
        self.proxies = torch.nn.Parameter(
            torch.randn(cfg["num_class"], cfg["num_bit"]).to(cfg["device"])
        )
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, x=None, y=None, label=None):
        P_one_hot = label

        cos = F.normalize(x, p=2, dim=1).mm(F.normalize(self.proxies, p=2, dim=1).T)
        pos = 1 - cos
        neg = F.relu(cos)

        cos_t = F.normalize(y, p=2, dim=1).mm(F.normalize(self.proxies, p=2, dim=1).T)
        pos_t = 1 - cos_t
        neg_t = F.relu(cos_t)

        P_num = len(P_one_hot.nonzero())
        N_num = len((P_one_hot == 0).nonzero())

        pos_term = (
            torch.where(
                P_one_hot == 1,
                pos.to(torch.float32),
                torch.zeros_like(cos).to(torch.float32),
            ).sum()
            / P_num
        )
        neg_term = (
            torch.where(
                P_one_hot == 0,
                neg.to(torch.float32),
                torch.zeros_like(cos).to(torch.float32),
            ).sum()
            / N_num
        )

        pos_term_t = (
            torch.where(
                P_one_hot == 1,
                pos_t.to(torch.float32),
                torch.zeros_like(cos_t).to(torch.float32),
            ).sum()
            / P_num
        )
        neg_term_t = (
            torch.where(
                P_one_hot == 0,
                neg_t.to(torch.float32),
                torch.zeros_like(cos_t).to(torch.float32),
            ).sum()
            / N_num
        )

        if 1 > 0:
            index = label.sum(dim=1) > 1
            label_ = label[index].float()

            x_ = x[index]
            t_ = y[index]

            cos_sim = label_.mm(label_.T)

            if len((cos_sim == 0).nonzero()) == 0:
                reg_term = 0
                reg_term_t = 0
                reg_term_xt = 0
            else:
                x_sim = F.normalize(x_, p=2, dim=1).mm(F.normalize(x_, p=2, dim=1).T)
                t_sim = F.normalize(t_, p=2, dim=1).mm(F.normalize(t_, p=2, dim=1).T)
                xt_sim = F.normalize(x_, p=2, dim=1).mm(F.normalize(t_, p=2, dim=1).T)

                neg = 0.1 * F.relu(x_sim)
                neg_t = 0.1 * F.relu(t_sim)
                neg_xt = 0.1 * F.relu(xt_sim)

                reg_term = torch.where(
                    cos_sim == 0, neg, torch.zeros_like(x_sim)
                ).sum() / len((cos_sim == 0).nonzero())
                reg_term_t = torch.where(
                    cos_sim == 0, neg_t, torch.zeros_like(t_sim)
                ).sum() / len((cos_sim == 0).nonzero())
                reg_term_xt = torch.where(
                    cos_sim == 0, neg_xt, torch.zeros_like(xt_sim)
                ).sum() / len((cos_sim == 0).nonzero())
        else:
            reg_term = 0
            reg_term_t = 0
            reg_term_xt = 0

        return (
            pos_term
            + neg_term
            + pos_term_t
            + neg_term_t
            + reg_term
            + reg_term_t
            + reg_term_xt
        )


class CONSTRASTIVE_LOSS(nn.Module):
    def __init__(self, batch_size=cfg["train_batch_size"], temperature=0.5):
        super(CONSTRASTIVE_LOSS, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.negatives_mask = 0.8

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )  # simi_mat: (2*bs, 2*bs)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs
        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature
        )  # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class CPF(nn.Module):
    def __init__(self):
        super(CPF, self).__init__()

        self.out_features = cfg["num_class"]
        self.in_features = cfg["num_bit"]

        self.weight = nn.Parameter(
            torch.FloatTensor(self.out_features, self.in_features).to(cfg["device"])
        )
        nn.init.xavier_uniform_(self.weight)

        self.ls_eps = 0.8

        self.tau = 0.3
        self.psi = 0.3
        self.sp = 1.3
        self.sn = 1.3
        self.mu = 1.0
        self.b = 2

    def forward(self, image, text, labels):
        one_hot = labels.to(cfg["device"])

        cosine = F.linear(F.normalize(image), F.normalize(self.weight))
        t_cosine = F.linear(F.normalize(text), F.normalize(self.weight))
        tp = ((cosine.clamp(min=0.0) * one_hot) * 2).sum() + self.b
        t_tp = ((t_cosine.clamp(min=0.0) * one_hot) * 2).sum() + self.b

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        lossp = (
            (1.0 - cosine) * torch.exp((1.0 - cosine) * self.sp).detach() * one_hot
        ).sum()
        t_lossp = (
            (1.0 - t_cosine) * torch.exp((1.0 - t_cosine) * self.sp).detach() * one_hot
        ).sum()

        mask = cosine > self.tau
        cosine = cosine[mask]
        lossn = (
            (cosine - self.psi)
            * torch.exp((cosine - self.mu) * self.sn).detach()
            * (1 - one_hot[mask])
        ).sum()

        t_mask = t_cosine > self.tau
        t_cosine = t_cosine[t_mask]
        t_lossn = (
            (t_cosine - self.psi)
            * torch.exp((t_cosine - self.mu) * self.sn).detach()
            * (1 - one_hot[t_mask])
        ).sum()

        loss = 1.0 - (tp) / (tp + lossp + lossn)
        t_loss = 1.0 - (t_tp) / (t_tp + t_lossp + t_lossn)

        return loss + t_loss


class Cross_modal_class_balance_loss(nn.Module):
    def __init__(self, bit, gamma=2.0, alpha=0.25):
        super(Cross_modal_class_balance_loss, self).__init__()
        self.hash_targets = self.get_hash_targets(cfg["num_class"], bit).to(
            cfg["device"]
        )
        self.multi_label_random_center = (
            torch.randint(2, (bit,)).float().to(cfg["device"])
        )
        self.balance_weight = torch.tensor(0.004).float().to(cfg["device"])
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def reszet_parameters(self):
        self.weight.data.fill_(0.5)

    def hloss(self, outputs, targets):

        if torch.min(targets) >= 0:
            targets = 2 * targets - 1

        hinge_loss = 1 - outputs * targets
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss.mean()

    def forward(self, u, y):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.hloss(u, hash_center)
        Q_loss = (u.abs() - 1).pow(3).mean()
        y = y.float().to(cfg["device"])

        balance_loss = self.balance_weight * ((y.mean(dim=0) - 0.5).abs().mean())

        return torch.sigmoid(self.weight) * (center_loss + 0.1 * Q_loss + balance_loss)

    def label2center(self, y):
        y = y.float().to(cfg["device"])
        center_sum = y @ self.hash_targets
        random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
        center_sum[center_sum == 0] = random_center[center_sum == 0]
        hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()
        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for _ in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    # print(c.min(), c.mean())
                    break
        return hash_targets


def sigmoid(x, temp=1.0):
    """Sigmoid function with temperature."""
    return 1.0 / (1.0 + torch.exp(-x / temp))

class SmoothAP(nn.Module):
    """PyTorch implementation of the Smooth-AP loss."""
    def __init__(self, anneal=1.0):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        """
        super(SmoothAP, self).__init__()
        self.anneal = anneal

    def forward(self, img, text, label):
        """
        Forward pass for SmoothAP loss.

        Args:
            img: Image features [bs, dim]
            text: Text features [bs, dim]
            label: Labels [bs, num_classes] (one-hot encoded)
        """
        # Normalize features
        img = F.normalize(img, p=2, dim=1)
        text = F.normalize(text, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(img, text.T)  # [bs, bs]

        # Compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(similarity_matrix.size(0)).to(similarity_matrix.device)
        mask = mask.unsqueeze(dim=0).repeat(similarity_matrix.size(0), 1, 1)

        # Compute the relevance scores
        sim_all_repeat = similarity_matrix.unsqueeze(dim=1).repeat(1, similarity_matrix.size(0), 1)
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask

        # Compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # Compute the mask for positive pairs
        pos_mask = label.mm(label.T)  # [bs, bs]
        pos_mask = pos_mask.unsqueeze(dim=1).repeat(1, similarity_matrix.size(0), 1)

        # Compute the relevance scores for positive pairs
        sim_pos_repeat = similarity_matrix.unsqueeze(dim=1).repeat(1, similarity_matrix.size(0), 1)
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 2, 1)
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask

        # Compute the rankings of the positive pairs
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        # Compute the SmoothAP loss
        ap = torch.zeros(1).to(similarity_matrix.device)
        for i in range(similarity_matrix.size(0)):
            pos_divide = torch.sum(sim_pos_rk[i] / (sim_all_rk[i]))
            ap = ap + (pos_divide / similarity_matrix.size(0))

        return (1 - ap)
    
# cvpr
def global_prompt_alignment_loss(out_1, out_2, temperature=0.07):
    # out_*: ND
    bz = out_1.size(0)
    targets = torch.arange(bz).type_as(out_1).long()

    scores = out_1.mm(out_2.t())
    scores /= temperature
    scores1 = scores.transpose(0, 1)
    loss0 = F.cross_entropy(scores, targets)
    loss1 = F.cross_entropy(scores1, targets)

    return 0.5 * (loss0 + loss1)


def bayesian_loss(a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
    # a: ND
    # b: MD
    # label_sim: NM
    s = 0.5 * torch.matmul(a, b.t()).clamp(min=-64, max=64)
    b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))
    return b_loss


def quantization_loss_2(hash_feature, B, K_bits):
    return (
        F.mse_loss(hash_feature, B, reduction="sum") / (hash_feature.shape[0]) / K_bits
    )


# cvpr end


def similarity_loss(a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):

    vartheta = 0.75
    threshold = 0.1
    similarity_function = "cosine"

    similarity = torch.cosine_similarity(a, b)

    if similarity_function == "euclidean":
        positive_similarity = similarity * label_sim
        negative_similarity = similarity * (1 - label_sim)
        max_value = float(512 * 2 * vartheta) ** 0.5
        negative_similarity = negative_similarity.clip(max=max_value)
        negative_similarity = (
            torch.tensor([max_value]).expand_as(negative_similarity).to(a.device)
            * (1 - label_sim)
            - negative_similarity
        )

        positive_loss = torch.pow(positive_similarity, 2).mean()
        negative_loss = torch.pow(negative_similarity, 2).mean()

        return similarity, positive_loss, negative_loss

    elif similarity_function == "cosine":
        similarity = similarity.clip(min=threshold).clip(max=1 - threshold)
        sim_loss = -label_sim * torch.log(similarity) - (1 - label_sim) * torch.log(
            1 - similarity
        )
        return similarity, torch.mean(sim_loss), torch.mean(sim_loss)


def soft_argmax_hash_loss(o):
    hash_loss = 1 - torch.pow(2 * o - 1, 2).mean()
    return hash_loss


def our_loss(image, text, labels):
    label_sim = (labels.matmul(labels.transpose(0, 1)) > 0).float()

    label_sim = label_sim.to(0)

    intra_similarity, intra_positive_loss, intra_negative_loss = similarity_loss(
        image, text, label_sim
    )
    inter_similarity_i, inter_positive_loss_i, inter_negative_loss_i = similarity_loss(
        image, image, label_sim
    )
    inter_similarity_t, inter_positive_loss_t, inter_negative_loss_t = similarity_loss(
        text, text, label_sim
    )

    quan_image_loss = soft_argmax_hash_loss(image)
    quan_text_loss = soft_argmax_hash_loss(text)

    intra_similarity_loss = intra_positive_loss + intra_negative_loss
    inter_similarity_loss = (
        inter_positive_loss_t
        + inter_positive_loss_i
        + inter_negative_loss_i
        + inter_negative_loss_t
    )
    t = inter_similarity_loss + intra_similarity_loss

    quan_loss = (quan_image_loss + quan_text_loss) / 2
    loss = t + 0.1 * quan_loss

    return loss


def rect(npoints, ndim):
    vec = np.random.randint(0, 2, size=(npoints, ndim))
    vec[vec == 0] = -1
    return vec


def noise(embeedings, noises):
    embeedings = np.nan_to_num(embeedings, nan=0.0, posinf=1e10, neginf=-1e10)
    if np.isnan(noises).any() or np.isinf(noises).any():
        noises = np.nan_to_num(noises, nan=0.0, posinf=1e10, neginf=-1e10)
    data_size = embeedings.shape[0]
    assgined_noise = dict(zip(range(data_size), noises))
    losses = np.zeros(shape=(data_size, data_size), dtype="float32")
    for i in range(data_size):
        fts = np.repeat(np.expand_dims(embeedings[i], axis=0), data_size, axis=0)
        diff = fts - noises
        l2 = np.linalg.norm(diff, axis=1)
        losses[i] = l2
    row_ind, col_ind = linear_sum_assignment(losses)
    for r, c in zip(row_ind, col_ind):
        assgined_noise[r] = noises[c]
    new_noise = np.empty(shape=noises.shape, dtype="float32")
    for i in range(data_size):
        new_noise[i] = assgined_noise[i]
    return new_noise


def noise_loss(img_rot, text_rot):
    batch_size_, code_length = img_rot.shape
    s_vector = rect(batch_size_, code_length)
    i_noises = noise(img_rot.cpu().detach().numpy(), s_vector)
    t_noises = noise(text_rot.cpu().detach().numpy(), s_vector)
    i_noises = torch.from_numpy(i_noises).float().to(cfg["device"])
    t_noises = torch.from_numpy(t_noises).float().to(cfg["device"])
    i_noise_loss = img_rot.mul(i_noises).sum(dim=-1).mean()
    t_noise_loss = text_rot.mul(t_noises).sum(dim=-1).mean()
    noise_loss = i_noise_loss + t_noise_loss
    return 0.1 * noise_loss


def quantization_Loss(hash_feature, B):
    return (
        F.mse_loss(hash_feature, B, reduction="sum")
        / (hash_feature.shape[0])
        / B.shape[1]
    )

    # bs, bit = all_feature_bs.shape
    # quantization_loss = torch.sum(torch.pow(all_feature_bs - feature_bs, 2)) / (
    # bs * bit
    # )
    # return quantization_loss


def multilabelsimilarity_loss(
    label, labels_train, hashrepresentations_batchsize, hashrepresentations_train
):
    batch_size = label.shape[0]
    num_train = labels_train.shape[0]
    label = label / torch.sqrt(torch.sum(torch.pow(label, 2), 1)).unsqueeze(1)
    labels_train = labels_train / torch.sqrt(
        torch.sum(torch.pow(labels_train, 2), 1)
    ).unsqueeze(1)
    hashrepresentations_batchsize = hashrepresentations_batchsize / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations_batchsize, 2), 1)
    ).unsqueeze(1)
    hashrepresentations_train = hashrepresentations_train / torch.sqrt(
        torch.sum(torch.pow(hashrepresentations_train, 2), 1)
    ).unsqueeze(1)
    labelsSimilarity = torch.matmul(label, labels_train.t())
    hashrepresentationsSimilarity = torch.relu(
        torch.matmul(hashrepresentations_batchsize, hashrepresentations_train.t())
    )
    loss = torch.sum(torch.pow(hashrepresentationsSimilarity - labelsSimilarity, 2)) / (
        num_train * batch_size
    )

    return loss


def info_nce_loss(out_1, out_2, temperature=0.07):
    bz = out_1.size(0)
    targets = torch.arange(bz).type_as(out_1).long()
    scores = out_1.mm(out_2.t())
    scores /= temperature
    scores1 = scores.transpose(0, 1)
    loss0 = F.cross_entropy(scores, targets)
    loss1 = F.cross_entropy(scores1, targets)
    return 0.5 * (loss0 + loss1)


def info_nce_loss_bmm(out_1, out_2, temperature=0.07):
    out_1 = out_1.permute(1, 0, 2)  # NLD
    out_2 = out_2.permute(1, 0, 2)  # NLD
    bz = out_1.size(0)
    sim = torch.bmm(out_1, out_2.permute(0, 2, 1))
    sim /= temperature
    word_num = sim.shape[1]
    sim_1 = rearrange(sim, "b n1 n2 -> (b n1) n2")
    sim_2 = rearrange(sim, "b n1 n2 -> (b n2) n1")
    targets = torch.arange(word_num).type_as(out_1).long().repeat(bz)
    loss_1 = F.cross_entropy(sim_1, targets)
    loss_2 = F.cross_entropy(sim_2, targets)
    return 0.5 * (loss_1 + loss_2)


def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
    s = 0.5 * torch.matmul(a, b.t()).clamp(min=-64, max=64)
    b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))
    return b_loss


def quantization_loss(self, hash_feature, B):
    return (
        F.mse_loss(hash_feature, B, reduction="sum")
        / (hash_feature.shape[0])
        / self.output_dim
    )
