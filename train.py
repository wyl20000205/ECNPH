import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from optimization import BHCH_Adam
from config import cfg
from dataloader import dataloader
from triplet.losses import TripletCustomMarginLoss, LowerBoundLoss, bit_var_loss
from triplet.methods import MetricLearningMethods
from triplet.miners.triplet_automargin_miner import TripletAutoParamsMiner
from pytorch_metric_learning import distances, reducers
from BHCH import BHCH, JNet, GCN, GCNL, GCNLI, GCNLT
from cluster import CTM, TCBlock
from loss import (
    DPDH_LOSS,
    DNPH_LOSS,
    CPF,
    quantization_Loss,
    multilabelsimilarity_loss,
    noise_loss,
    our_loss,
    Cross_modal_class_balance_loss,
    DSPH_LOSS,
    SmoothAP,
)
from neighbor.n_main import *
from metrics import *
from rot import Rot
from load_data import generate_dataset

logging.basicConfig(
    filename="./log.txt", level=logging.INFO, format="%(asctime)s - %(message)s"
)


def get_config():
    parser = argparse.ArgumentParser(description="AdaTriplet")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--method", type=str, default="AdaTriplet-AM")
    parser.add_argument("--type_of_triplets", type=str, default="semihard")
    parser.add_argument("--dataset", type=str, default="mirflickr25k")
    parser.add_argument("--bit", type=int, default=64)

    return parser.parse_args()


class CentralityWeightingLoss(nn.Module):
    """
    Implementation of Centrality Weighting Loss (L_Wti) as described in Section 4.2.
    This loss emphasizes the learning of hubs by incorporating centrality weights
    into the contrastive loss function.
    """

    def __init__(self, config=None):
        super(CentralityWeightingLoss, self).__init__()

    def forward(self, similarity_matrix, centrality_weights):
        """
        Compute Centrality Weighting Loss.

        Args:
            similarity_matrix: Cross-modal similarity matrix
            centrality_weights: Weights based on centrality scores (Eq. 3 in paper)

        Returns:
            Centrality Weighting Loss value
        """
        # Compute log probabilities (softmax over rows)
        log_probabilities = F.log_softmax(similarity_matrix, dim=-1)

        # Extract diagonal elements (matching pairs)
        diagonal_log_probs = torch.diag(log_probabilities)

        # Apply centrality weights (Eq. 4 in paper)
        weighted_log_probs = diagonal_log_probs * centrality_weights

        # Compute negative log likelihood
        nce_loss = -weighted_log_probs

        # Return mean loss
        loss = nce_loss.mean()
        return loss


class NeighborAdjustingLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()

    def normalize_similarity(self, sim, mask):
        masked_min = torch.where(mask == 0, sim, 9e15)
        masked_max = torch.where(mask == 0, sim, -9e15)
        minv = masked_min.min(dim=-1, keepdim=True)[0]
        maxv = masked_max.max(dim=-1, keepdim=True)[0]
        return (sim - minv) / (maxv - minv + 1e-8)

    def create_neighbor_mask(self, sim, k):
        B = sim.size(0)
        device = sim.device
        eye = torch.eye(B, device=device)
        sim_no_self = torch.where(eye == 0, sim, -9e15)
        _, idx = torch.sort(sim_no_self, dim=-1, descending=True)
        top_k = idx[:, :k]  # (B,k)
        nb_mask = torch.zeros_like(sim).scatter_(1, top_k, 1.0)
        ext_mask = eye + nb_mask  # 对角+邻居
        return nb_mask, ext_mask

    def compute_positive_weights(self, sim, nb_mask, T):
        pos_w = torch.softmax(sim * T, dim=-1)
        pos_w = pos_w * nb_mask
        pos_w.fill_diagonal_(1.0)
        return pos_w

    def forward(self, sim_mat, mb_mat, num_neighbors, temperature):
        """
        sim_mat : (B,B)   当前 batch 图文相似度
        mb_mat  : (B,M)   memory-bank 相似度（用于估计中心性）
        """
        device = sim_mat.device
        B = sim_mat.size(0)
        k = min(num_neighbors, B - 1)  # 防越界
        nb_mask, ext_mask = self.create_neighbor_mask(sim_mat, k)

        # 中心性估计：memory-bank 行平均
        centrality = mb_mat.mean(dim=-1)  # (B,)
        centrality = centrality.unsqueeze(0).repeat(B, 1)  # (B,B)

        # 归一化
        norm_sim = self.normalize_similarity(sim_mat, ext_mask)
        norm_cen = self.normalize_similarity(centrality, ext_mask)

        # de-centrality 修正
        adj_sim = torch.where(nb_mask == 1, norm_sim - norm_cen, -9e15)

        # 正样本权重
        pos_w = self.compute_positive_weights(adj_sim, nb_mask, temperature)

        # 加权 log-softmax
        masked_sim = torch.where(ext_mask == 1, sim_mat, -9e15)
        log_p = F.log_softmax(masked_sim, dim=-1) * pos_w
        loss = -(log_p.sum(dim=-1) / pos_w.sum(dim=-1)).mean()
        return loss


class UniformRegularizationLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()

    @torch.no_grad()
    def sinkhorn_algorithm(self, scores, beta=0.3, num_iters=50):
        """
        在 log-space 做 Sinkhorn，返回 transport plan Q
        scores: (B,B) 相似度矩阵
        """
        m, n = scores.shape
        device = scores.device
        ms = torch.tensor(m, device=device, dtype=scores.dtype)
        ns = torch.tensor(n, device=device, dtype=scores.dtype)

        # 初始对偶变量
        log_mu = -(ms + ns).log().expand(m)
        log_nu = -(ms + ns).log().expand(n)
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)

        # 迭代
        for _ in range(num_iters):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(0), dim=1)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(1), dim=0)

        # log transport plan
        Z = scores + u.unsqueeze(1) + v.unsqueeze(0) - (ms + ns).log()
        Q = Z.exp()  # (B,B)  双随机矩阵

        # 插值
        I = torch.eye(m, device=device)
        targets = beta * Q + (1 - beta) * I
        return targets

    def forward(self, similarity_matrix, logit_scale=1.0, beta=0.3, num_iterations=50):
        """
        similarity_matrix: (B,B)  图文相似度（已除 temp）
        logit_scale:       float   额外缩放，默认 1 即可
        beta:              float   OT 与 identity 混合权重
        num_iterations:    int     Sinkhorn 迭代次数
        """
        # 最优传输计划
        targets = self.sinkhorn_algorithm(
            similarity_matrix, beta=beta, num_iters=num_iterations
        )  # (B,B)

        # 加权交叉熵  (Eq.11 & 12)
        log_prob = F.log_softmax(similarity_matrix * logit_scale, dim=-1)  # (B,B)
        loss = -(targets * log_prob).sum(dim=-1).mean()
        return loss


class KLDivergenceLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()

    def forward(self, global_similarity, local_similarity):
        """
        global_similarity: (B,B)  全局相似度矩阵
        local_similarity : (B,B)  局部相似度矩阵
        """
        # 全局分布 P
        global_log_probs = F.log_softmax(global_similarity, dim=-1)  # (B,B)
        # 局部分布 Q
        local_probs = F.softmax(local_similarity, dim=-1)  # (B,B)

        # KL(Q‖P) = ∑ Q·(log Q - log P)
        kl_loss = F.kl_div(global_log_probs, local_probs, reduction="mean")
        return kl_loss


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = 0
        ctx.batch_size = tensor.shape[0]
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )


allgather = AllGather.apply


class WYL_Trainer:
    def __init__(self):
        self._init_memory_bank()
        self._init_loss_functions()
        self._init_token_clustering()
        self._init_weighting_networks()
        self.initialize_memory_bank(feature_dim=64, capacity=2048, device=cfg["device"])
        self.kl_weight = 1.0
        self.uniform_weight = 1.0
        self.neighbor_weight = 1.0
        self.centrality_scale = 0.3
        self.temperature = 0.1
        self.beta = 0.7
        self.num_neighbors = 20
        self.model = BHCH().float().to(cfg["device"])
        self.logit_scale = self.model.clip.logit_scale.exp()
        self.dsph = DSPH_LOSS().to(cfg["device"])
        self.model.float()

        self.all_dataloader, self.all_label = self._init_dataset()
        self.train_loader, self.query_loader, self.retrieval_loader = (
            self.all_dataloader
        )
        self.train_labels, self.query_labels, self.retrieval_labels = self.all_label
        self.train_labels, self.query_labels, self.retrieval_labels = (
            self.train_labels.to(cfg["device"]),
            self.query_labels.to(cfg["device"]),
            self.retrieval_labels.to(cfg["device"]),
        )
        self.num_train, self.num_query, self.num_retrieval = (
            self.train_labels.shape[0],
            self.query_labels.shape[0],
            self.retrieval_labels.shape[0],
        )
        self.train_log = f"dataset:{cfg['dataset']} class:{self.train_labels.shape[1]} dropout:{cfg['dropout']} train:{self.num_train} query:{self.num_query} retrieval:{self.num_retrieval} total:{self.num_query+self.num_retrieval}"
        self.optimizer = BHCH_Adam(
            [
                {"params": self.model.clip.parameters(), "lr": cfg["clip_lr"]},
                {"params": self.model.image_pre.parameters(), "lr": cfg["other_lr"]},
                {"params": self.model.text_pre.parameters(), "lr": cfg["other_lr"]},
            ],
            lr=0.001,
            warmup=0.1,
            schedule="warmup_cosine",
            b1=0.9,
            b2=0.98,
            e=1e-6,
            t_total=len(self.train_loader) * cfg["train_epoch"],
            weight_decay=0.2,
            max_grad_norm=1.0,
        )
        self.optimizer_dsph = torch.optim.SGD(
            params=self.dsph.parameters(), lr=0.00001, weight_decay=0.0005
        )

    def train(self):
        print(self.train_log)
        logging.info(f"\n\n{self.train_log}")
        self.model.train()
        for i in range(cfg["train_epoch"]):
            for step, (image, text, key_padding_mask, label, index) in enumerate(
                tqdm(self.train_loader)
            ):
                image, text, key_padding_mask, label = (
                    image.to(cfg["device"], non_blocking=True),
                    text.to(cfg["device"], non_blocking=True),
                    key_padding_mask.to(cfg["device"], non_blocking=True),
                    label.to(cfg["device"], non_blocking=True).float(),
                )
                # index = index.numpy()
                image_features, text_features, img_hash, text_hash = self.model(
                    image, text, label
                )

                global_it = F.normalize(image_features @ text_features.t(), p=2, dim=-1)
                global_ii = F.normalize(
                    image_features @ image_features.t(), p=2, dim=-1
                )
                global_tt = F.normalize(text_features @ text_features.t(), p=2, dim=-1)
                local_it = F.normalize(img_hash @ text_hash.t(), p=2, dim=-1)
                local_ii = F.normalize(img_hash @ img_hash.t(), p=2, dim=-1)
                local_tt = F.normalize(text_hash @ text_hash.t(), p=2, dim=-1)
                aa = F.log_softmax(global_it, dim=-1)
                bb = F.softmax(local_it, dim=-1)
                cc = F.log_softmax(global_ii, dim=-1)
                dd = F.softmax(local_ii, dim=-1)
                ee = F.log_softmax(global_tt, dim=-1)
                ff = F.softmax(local_tt, dim=-1)

                kl_loss = (
                    F.kl_div(aa, ff, reduction="batchmean")
                    + F.kl_div(cc, dd, reduction="batchmean")
                    + F.kl_div(ee, ff, reduction="batchmean")
                ) / 3

                diagonal_a = torch.diag(global_it).sum().item()
                diagonal_b = torch.diag(global_ii).sum().item()
                diagonal_c = torch.diag(global_tt).sum().item()
                diagonal_d = torch.diag(local_ii).sum().item()
                diagonal_e = torch.diag(local_tt).sum().item()
                diagonal_f = torch.diag(local_ii).sum().item()
                diagonal_loss = (
                    diagonal_a
                    + diagonal_b
                    + diagonal_c
                    + diagonal_d
                    + diagonal_e
                    + diagonal_f
                ) / 6

                text_feat, image_feat, text_mask, image_mask = self.generate_mask(
                    cfg["train_batch_size"], img_hash, text_hash
                )
                with torch.no_grad():
                    self.update_memory_bank(
                        index, text_feat, image_feat, text_mask, image_mask
                    )
                index = allgather(index)
                text_feat = allgather(text_feat)
                image_feat = allgather(image_feat)
                text_mask = allgather(text_mask)
                image_mask = allgather(image_mask)
                mb_feat_t = self.mb_feat_t
                mb_feat_v = self.mb_feat_v
                mb_mask_t = self.mb_mask_t
                mb_mask_v = self.mb_mask_v

                local_t2v_logits, local_v2t_logits = self.local_level(
                    text_feat, image_feat, text_mask, image_mask
                )

                (
                    uniform_loss,
                    global_text_feat,
                    global_image_feat,
                    global_t2v_logits,
                    global_v2t_logits,
                ) = self.compute_uniform_loss(
                    text_feat,
                    image_feat,
                    text_mask,
                    image_mask,
                    self.temperature,
                    self.beta,
                )
                kl_loss_1 = (
                    self.kl_loss(global_t2v_logits, local_t2v_logits)
                    + self.kl_loss(global_v2t_logits, local_v2t_logits)
                ) / 2
                centrality_loss = self.compute_centrality_loss(
                    text_feat,
                    image_feat,
                    global_text_feat,
                    global_image_feat,
                    local_t2v_logits,
                    local_v2t_logits,
                    self.centrality_scale,
                    self.logit_scale,
                )

                neighbor_loss = self.compute_neighbor_loss(
                    text_feat,
                    image_feat,
                    text_mask,
                    image_mask,
                    mb_feat_t,
                    mb_feat_v,
                    mb_mask_t,
                    mb_mask_v,
                    local_t2v_logits,
                    local_v2t_logits,
                    self.num_neighbors,
                    self.temperature,
                )
                # loss = (
                #     centrality_loss  # 有问题
                #     + (uniform_loss * self.uniform_weight)
                #     + (neighbor_loss * self.neighbor_weight)
                #     + (kl_loss * self.kl_weight)
                # )
                loss = (
                    kl_loss * 1000
                    + diagonal_loss * 0.1
                    + (uniform_loss * self.uniform_weight)
                    # +(neighbor_loss * self.neighbor_weight)
                    # + (kl_loss * self.kl_weight)
                    + self.dsph(img_hash, text_hash, label)
                )
                # loss = (uniform_loss * self.uniform_weight) + self.dsph(
                #     img_hash, text_hash, label
                # )
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_dsph.zero_grad(set_to_none=True)
                loss.backward()  # 解决 centrality_loss
                self.optimizer.step()
                self.optimizer_dsph.step()
            # self.valid(i, loss.item())
            logging.info(f"{1}")

    def valid(self, epoch, all_loss):
        self.model.eval()
        query_i_buffer, query_t_buffer = self.encode(self.query_loader, self.num_query)
        retrieval_i_buffer, retrieval_t_buffer = self.encode(
            self.retrieval_loader, self.num_retrieval
        )
        mAPi2t = calc_map_k_matrix(
            query_i_buffer,
            retrieval_t_buffer,
            self.query_labels,
            self.retrieval_labels,
        )
        mAPt2i = calc_map_k_matrix(
            query_t_buffer,
            retrieval_i_buffer,
            self.query_labels,
            self.retrieval_labels,
        )
        print(
            f"{str(epoch).zfill(2)}/{cfg['train_epoch']} {cfg['num_bit']}bit all_loss:{all_loss:.4f} mAPi2t:{mAPi2t:.4f} mAPt2i:{mAPt2i:.4f}"
        )
        logging.info(
            f"{str(epoch).zfill(2)}/{cfg['train_epoch']} {cfg['num_bit']}bit all_loss:{all_loss:.4f} mAPi2t:{mAPi2t:.4f} mAPt2i:{mAPt2i:.4f}"
        )
        return mAPi2t, mAPt2i, 1, 1

    def encode(self, data_loader, length):
        img_buffer = torch.empty(length, cfg["num_bit"], dtype=torch.float).to(
            cfg["device"]
        )
        text_buffer = torch.empty(length, cfg["num_bit"], dtype=torch.float).to(
            cfg["device"]
        )
        for image, text, padding_mask, label, index in tqdm(data_loader):
            image = image.to(cfg["device"], non_blocking=True)
            text = text.to(cfg["device"], non_blocking=True)
            label = label.to(cfg["device"], non_blocking=True).float()
            padding_mask = padding_mask.to(cfg["device"], non_blocking=True)
            index = index.numpy()
            image_features, text_features, image_hash, text_hash = self.model(
                image, text, label
            )
            image_hash = torch.sign(image_hash)
            text_hash = torch.sign(text_hash)
            # print(img_buffer.shape, image_hash.shape)
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data
        return img_buffer, text_buffer

    def _init_dataset(self):
        index_file = os.path.join(
            cfg["project_root"], "data_mat", cfg["dataset"], cfg["index_file"]
        )
        caption_file = os.path.join(
            cfg["project_root"], "data_mat", cfg["dataset"], cfg["caption_file"]
        )
        label_file = os.path.join(
            cfg["project_root"], "data_mat", cfg["dataset"], cfg["label_file"]
        )
        train_data, query_data, retrieval_data = generate_dataset(
            captionFile=caption_file,
            indexFile=index_file,
            labelFile=label_file,
            dataset_name=cfg["dataset"],
            maxWords=cfg["max_words"],
            imageResolution=cfg["image_resolution"],
            query_num=cfg["num_query"],
            train_num=cfg["num_train"],
            seed=cfg["seed"],
        )

        train_labels = train_data.get_all_label().float()
        query_labels = query_data.get_all_label().float()
        retrieval_labels = retrieval_data.get_all_label().float()

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=cfg["train_batch_size"],
            num_workers=cfg["num_workers"],
            pin_memory=True,
            shuffle=True,
            prefetch_factor=2,
        )
        query_loader = DataLoader(
            dataset=query_data,
            batch_size=cfg["query_batch_size"],
            num_workers=cfg["num_workers"],
            pin_memory=True,
            shuffle=True,
            prefetch_factor=2,
        )
        retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=cfg["retrieval_batch_size"],
            num_workers=cfg["num_workers"],
            pin_memory=True,
            shuffle=True,
            prefetch_factor=2,
        )

        return (train_loader, query_loader, retrieval_loader), (
            train_labels,
            query_labels,
            retrieval_labels,
        )

    def _init_loss_functions(self):
        self.centrality_weighting_loss = CentralityWeightingLoss()
        self.neighbor_adjusting_loss = NeighborAdjustingLoss()
        self.uniform_regularization_loss = UniformRegularizationLoss()
        self.kl_loss = KLDivergenceLoss()

    def _init_memory_bank(self):
        device = torch.device(cfg["device"])  # Will be moved to correct device later

        # Create empty tensors for the memory bank
        self.mb_ind = torch.tensor([], dtype=torch.long, device=device)
        self.mb_feat_t = torch.empty(
            (1, 1, 64), dtype=torch.float, device=device
        )  # 初始化为至少有1个样本
        self.mb_feat_v = torch.empty(
            (1, 1, 64), dtype=torch.float, device=device
        )  # 初始化为至少有1个样本
        self.mb_mask_t = torch.empty((1, 1), dtype=torch.float, device=device)
        self.mb_mask_v = torch.empty((1, 1), dtype=torch.float, device=device)
        self.mb_batch = 0

    def _init_weighting_networks(self):
        # Token weighting networks
        self.text_weight_fc = self._create_weighting_network().to(cfg["device"])
        self.video_weight_fc = self._create_weighting_network().to(cfg["device"])
        self.text_weight_fc0 = self._create_weighting_network().to(cfg["device"])
        self.video_weight_fc0 = self._create_weighting_network().to(cfg["device"])
        self.text_weight_fc1 = self._create_weighting_network().to(cfg["device"])
        self.video_weight_fc1 = self._create_weighting_network().to(cfg["device"])
        self.text_weight_intra = self._create_weighting_network().to(cfg["device"])
        self.video_weight_intra = self._create_weighting_network().to(cfg["device"])

    def _create_weighting_network(self):
        return nn.Sequential(
            nn.Linear(64, 2 * 64), nn.ReLU(inplace=True), nn.Linear(2 * 64, 1)
        )

    def initialize_memory_bank(self, feature_dim, capacity=2048, device=None):
        """
        初始化一个固定容量的记忆库，用于存储全局文本和图像特征。
        Args:
            feature_dim: 特征维度（例如64）
            capacity: 最大存储样本数量
            device: 存储设备
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_bank = {
            "text": torch.zeros(capacity, feature_dim, device=device),
            "image": torch.zeros(capacity, feature_dim, device=device),
            "ptr": 0,  # 环形索引指针
            "size": 0,  # 当前存储样本数
            "capacity": capacity,
        }

    def _init_token_clustering(self):
        # Text token merging layers
        self.text_ctm0 = CTM(sample_ratio=1 / 6, embed_dim=64, dim_out=64, k=15).to(
            cfg["device"]
        )
        self.text_block0 = TCBlock(dim=64, num_heads=8).to(cfg["device"])
        self.text_ctm1 = CTM(sample_ratio=1 / 4, embed_dim=64, dim_out=64, k=15).to(
            cfg["device"]
        )
        self.text_block1 = TCBlock(dim=64, num_heads=8).to(cfg["device"])

        # Video token merging layers
        self.video_ctm0 = CTM(sample_ratio=1 / 4, embed_dim=64, dim_out=64, k=15).to(
            cfg["device"]
        )
        self.video_block0 = TCBlock(dim=64, num_heads=8).to(cfg["device"])
        self.video_ctm1 = CTM(sample_ratio=1 / 3, embed_dim=64, dim_out=64, k=15).to(
            cfg["device"]
        )
        self.video_block1 = TCBlock(dim=64, num_heads=8).to(cfg["device"])

    def update_memory_bank(self, idx, text_feat, image_feat, text_mask, image_mask):
        idx = idx.to(cfg["device"])
        if self.mb_feat_v.size(0) == 0:
            self.mb_ind = idx.clone()
            self.mb_feat_v = image_feat.clone()
            self.mb_feat_t = text_feat.clone()
            self.mb_mask_t = text_mask.clone()
            self.mb_mask_v = image_mask.clone()
            self.mb_batch = idx.size(0)
        else:
            # 拼接新样本到记忆库
            self.mb_ind = torch.cat((idx, self.mb_ind), dim=0)
            self.mb_feat_v = torch.cat((image_feat, self.mb_feat_v), dim=0)
            self.mb_feat_t = torch.cat((text_feat, self.mb_feat_t), dim=0)
            self.mb_mask_t = torch.cat((text_mask, self.mb_mask_t), dim=0)
            self.mb_mask_v = torch.cat((image_mask, self.mb_mask_v), dim=0)

        # 维护记忆库大小 (FIFO queue)
        mb_capacity = self.mb_feat_v.size(0)
        if self.mb_ind.size(0) > mb_capacity:
            self.mb_ind = self.mb_ind[:mb_capacity]
            self.mb_feat_v = self.mb_feat_v[:mb_capacity]
            self.mb_feat_t = self.mb_feat_t[:mb_capacity]
            self.mb_mask_t = self.mb_mask_t[:mb_capacity]
            self.mb_mask_v = self.mb_mask_v[:mb_capacity]

    def generate_mask(self, bs, text_vec, img_vec):
        device = cfg["device"]
        text_feat = text_vec.unsqueeze(1)  # (bs, 1, 64)
        image_feat = img_vec.unsqueeze(1)  # (bs, 1, 64)
        text_mask = torch.ones((bs, 1), dtype=torch.long, device=device)
        image_mask = torch.ones((bs, 1), dtype=torch.long, device=device)
        return text_feat, image_feat, text_mask, image_mask

    def merge_global_features(self, text_feat, image_feat, text_mask, image_mask):
        # Prepare token dictionaries for text
        t_idx_token = torch.arange(text_feat.size(1))[None, :].repeat(
            text_feat.size(0), 1
        )
        t_agg_weight = text_feat.new_ones(text_feat.size(0), text_feat.size(1), 1)
        t_token_dict = {
            "x": text_feat,
            "token_num": text_feat.size(1),
            "idx_token": t_idx_token,
            "agg_weight": t_agg_weight,
            "mask": text_mask.detach(),
        }

        # Prepare token dictionaries for image
        v_idx_token = torch.arange(image_feat.size(1))[None, :].repeat(
            image_feat.size(0), 1
        )
        v_agg_weight = image_feat.new_ones(image_feat.size(0), image_feat.size(1), 1)
        v_token_dict = {
            "x": image_feat,
            "token_num": image_feat.size(1),
            "idx_token": v_idx_token,
            "agg_weight": v_agg_weight,
            "mask": image_mask.detach(),
        }

        # First level of token merging
        t_token_dict = self.text_block0(self.text_ctm0(t_token_dict))
        v_token_dict = self.video_block0(self.video_ctm0(v_token_dict))
        text_feat = t_token_dict["x"]
        image_feat = v_token_dict["x"]

        # Second level of token merging
        t_token_dict = self.text_block1(self.text_ctm1(t_token_dict))
        v_token_dict = self.video_block1(self.video_ctm1(v_token_dict))
        text_feat = t_token_dict["x"]
        image_feat = v_token_dict["x"]

        return text_feat, image_feat

    def compute_centrality_loss(
        self,
        text_feat,
        image_feat,
        global_text_feat,
        global_image_feat,
        local_t2v_logits,
        local_v2t_logits,
        centrality_scale,
        logit_scale,
    ):
        local_text_weights, local_img_weights = self.compute_centrality_weights(
            text_feat, image_feat, global_text_feat, global_image_feat, centrality_scale
        )

        centrality_loss_t2v = self.centrality_weighting_loss(
            local_t2v_logits * logit_scale, local_text_weights
        )
        centrality_loss_v2t = self.centrality_weighting_loss(
            local_v2t_logits * logit_scale, local_img_weights
        )

        return (centrality_loss_t2v + centrality_loss_v2t) / 2

    def compute_centrality_weights(
        self,
        text_feat,
        image_feat,
        global_text_feat,
        global_image_feat,
        centrality_scale,
    ):
        """
        单 token 场景：
          text_feat        : (B,1,64)  ← 来自 clip_to_merge
          image_feat       : (B,1,64)
          global_text_feat : (B,1,64)  ← 可再经过一层 attention 后的全局向量
          global_image_feat: (B,1,64)
          centrality_scale : float
        返回：
          text_weights  : (B,)
          image_weights : (B,)
        """
        # 直接 squeeze 掉单 token 维度
        text_local = text_feat.squeeze(1)  # (B,64)
        image_local = image_feat.squeeze(1)  # (B,64)
        text_global = global_text_feat.squeeze(1)  # (B,64)
        image_global = global_image_feat.squeeze(1)  # (B,64)

        text_local = F.normalize(text_local, dim=-1)
        image_local = F.normalize(image_local, dim=-1)
        text_global = F.normalize(text_global, dim=-1)
        image_global = F.normalize(image_global, dim=-1)

        # 全局-局部相似度 → 这里局部只有 1 个 token，所以退化成自己跟自己算 cos
        text_centrality = torch.sum(text_global * text_local, dim=1)  # (B,)
        image_centrality = torch.sum(image_global * image_local, dim=1)  # (B,)

        # 指数缩放得权重
        text_weights = torch.exp(text_centrality * centrality_scale)
        image_weights = torch.exp(image_centrality * centrality_scale)

        return text_weights, image_weights

    def compute_neighbor_loss(
        self,
        text_feat,
        image_feat,
        text_mask,
        image_mask,
        mb_feat_t,
        mb_feat_i,
        mb_mask_t,
        mb_mask_i,
        local_t2v_logits,
        local_v2t_logits,
        num_neighbors,
        temperature,
    ):
        memory_bank_t2v_logits, _ = self.local_level(
            text_feat, mb_feat_i, text_mask, mb_mask_i
        )
        _, memory_bank_v2t_logits = self.local_level(
            mb_feat_t, image_feat, mb_mask_t, image_mask
        )

        neighbor_loss_t2v = self.neighbor_adjusting_loss(
            local_t2v_logits, memory_bank_v2t_logits, num_neighbors, temperature
        )
        neighbor_loss_v2t = self.neighbor_adjusting_loss(
            local_v2t_logits, memory_bank_t2v_logits, num_neighbors, temperature
        )

        return (neighbor_loss_t2v + neighbor_loss_v2t) / 2

    def compute_uniform_loss(
        self, text_feat, image_feat, text_mask, image_mask, temperature, beta
    ):
        global_text_feat, global_image_feat = self.merge_global_features(
            text_feat, image_feat, text_mask, image_mask
        )
        global_t2v_logits, global_v2t_logits = self.global_level(
            global_text_feat, global_image_feat
        )
        uniform_loss_t2v = self.uniform_regularization_loss(
            global_t2v_logits, temperature, beta
        )
        uniform_loss_v2t = self.uniform_regularization_loss(
            global_v2t_logits, temperature, beta
        )
        uniform_loss = (uniform_loss_t2v + uniform_loss_v2t) / 2
        return (
            uniform_loss,
            global_text_feat,
            global_image_feat,
            global_t2v_logits,
            global_v2t_logits,
        )

    def local_level(self, text_feat, image_feat, text_mask, image_mask):
        # Calculate attention weights for text tokens
        text_weight = self.text_weight_fc(text_feat).squeeze(2)  # [B, N_t]
        text_weight.masked_fill_((1 - text_mask).to(torch.bool), float(-9e15))
        text_weight = torch.softmax(text_weight, dim=-1)  # [B, N_t]

        # Calculate attention weights for video tokens
        video_weight = self.video_weight_fc(image_feat).squeeze(2)  # [B, N_v]
        video_weight.masked_fill_((1 - image_mask).to(torch.bool), float(-9e15))
        video_weight = torch.softmax(video_weight, dim=-1)  # [B, N_v]

        # Normalize features
        text_feat = F.normalize(text_feat, dim=-1)
        image_feat = F.normalize(image_feat, dim=-1)

        # Calculate similarity between all text and video tokens
        retrieve_logits = torch.einsum("atd,bvd->abtv", [text_feat, image_feat])
        retrieve_logits = torch.einsum("abtv,at->abtv", [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum("abtv,bv->abtv", [retrieve_logits, image_mask])

        # Calculate text-to-video similarity
        t2v_logits, _ = retrieve_logits.max(dim=-1)  # [B, B, N_t]
        t2v_similarity = torch.einsum("abt,at->ab", [t2v_logits, text_weight])  # [B, B]

        # Calculate video-to-text similarity
        v2t_logits, _ = retrieve_logits.max(dim=-2)  # [B, B, N_v]
        v2t_similarity = torch.einsum(
            "abv,bv->ab", [v2t_logits, video_weight]
        )  # [B, B]

        # Average bidirectional similarities
        similarity = (t2v_similarity + v2t_similarity) / 2.0

        return similarity, similarity.T

    def global_level(self, text_feat, image_feat):
        """
        输入单 token 全局特征：
        text_feat : (B,1,64)
        image_feat: (B,1,64)
        输出 (B,B) 相似度矩阵
        """
        # ---- attention weight ----
        text_w = torch.softmax(
            self.text_weight_fc1(text_feat).squeeze(-1), dim=1
        )  # (B,1)
        video_w = torch.softmax(
            self.video_weight_fc1(image_feat).squeeze(-1), dim=1
        )  # (B,1)

        # ---- 归一化 ----
        text_feat = F.normalize(text_feat, dim=-1)
        image_feat = F.normalize(image_feat, dim=-1)

        # ---- 全局相似度 ----
        text_vec = text_feat.squeeze(1)  # (B,64)
        image_vec = image_feat.squeeze(1)  # (B,64)
        similarity = torch.matmul(text_vec, image_vec.t())  # (B,B)

        return similarity, similarity.t()


if __name__ == "__main__":
    WYL_Trainer().train()
