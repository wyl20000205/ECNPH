import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from optimization import DPDH_Adam
from config import cfg
from dataloader import dataloader
from triplet.losses import TripletCustomMarginLoss, LowerBoundLoss, bit_var_loss
from triplet.methods import MetricLearningMethods
from triplet.miners.triplet_automargin_miner import TripletAutoParamsMiner
from pytorch_metric_learning import distances, reducers
from BHCH import DPDH_Encoder, JNet, GCN, GCNL, GCNLI, GCNLT
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


class AllGather2(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor with improved gradient handling."""

    # https://github.com/PyTorchLightning/lightning-bolts/blob/8d3fbf7782e3d3937ab8a1775a7092d7567f2933/pl_bolts/models/self_supervised/simclr/simclr_module.py#L20
    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False
        )
        return (
            grad_input[ctx.rank * ctx.batch_size : (ctx.rank + 1) * ctx.batch_size],
            None,
        )


allgather = AllGather.apply
allgather2 = AllGather2.apply


class WYL_Trainer:
    def __init__(self):
        self._init_loss_functions()
        self._init_memory_bank()
        self.kl_weight = 1.0
        self.uniform_weight = 1.0
        self.neighbor_weight = 1.0
        self.logit_scale = self.model.clip.logit_scale.exp()
        self.centrality_scale = 0.3
        self.temperature = 0.1
        self.beta = 0.7
        self.num_neighbors = 20

        self.model = DPDH_Encoder().float().to(cfg["device"])
        self.dpdh = DPDH_LOSS().to(cfg["device"])
        self.cpf = CPF().to(cfg["device"])
        self.dnph = DNPH_LOSS().to(cfg["device"])
        self.dsph = DSPH_LOSS().to(cfg["device"])
        # self.CodeNet_J = JNet(code_len=cfg["num_bit"]).to(cfg["device"])
        # self.gcn_I = GCNLI(code_len=cfg["num_bit"]).to(cfg["device"])
        # self.gcn_T = GCNLT(code_len=cfg["num_bit"]).to(cfg["device"])
        # self.gcn_L = GCNL()
        self.model.float()
        self.banlance_loss = Cross_modal_class_balance_loss(cfg["num_bit"]).to(
            cfg["device"]
        )
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
        self.optimizer = DPDH_Adam(
            [
                {"params": self.model.clip.parameters(), "lr": cfg["clip_lr"]},
                {"params": self.model.image_pre.parameters(), "lr": cfg["other_lr"]},
                {"params": self.model.text_pre.parameters(), "lr": cfg["other_lr"]},
                # {"params": self.CodeNet_J.parameters(), "lr": cfg["other_lr"]},
                # {"params": self.gcn_I.parameters(), "lr": cfg["other_lr"]},
                # {"params": self.gcn_T.parameters(), "lr": cfg["other_lr"]},
                # {"params": self.gcn_L.parameters(), "lr": cfg["other_lr"]},
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
        # self.opt_GI = torch.optim.Adam(self.gcn_I.parameters(), lr=0.001)
        # self.opt_GT = torch.optim.Adam(self.gcn_T.parameters(), lr=0.001)
        # self.opt_GL = torch.optim.Adam(self.gcn_L.parameters(), lr=0.0001)
        # self.optimizer_loss = torch.optim.SGD(
        #     params=self.dpdh.parameters(), lr=0.00001, weight_decay=0.0005
        # )
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
                index = index.numpy()
                image_features, text_features, img_hash, text_hash = self.model(
                    image, text, label
                )
                text_feat = allgather(text_feat)
                video_feat = allgather(video_feat)
                text_mask = allgather(text_mask)
                video_mask = allgather(video_mask)
                torch.distributed.barrier()
                mb_feat_t = self.mb_feat_t
                mb_feat_v = self.mb_feat_v
                mb_mask_t = self.mb_mask_t
                mb_mask_v = self.mb_mask_v
                text_feat, img_feat, text_mask, img_mask = self.reshape_feat_gener_mask(
                    image_features, text_features, key_padding_mask
                )
                local_t2v_logits, local_v2t_logits = self.local_level(
                    text_feat, img_feat, text_mask, img_mask
                )
                (
                    uniform_loss,
                    global_text_feat,
                    global_img_feat,
                    global_t2v_logits,
                    global_v2t_logits,
                ) = self.compute_uniform_loss(
                    text_feat,
                    img_feat,
                    text_mask,
                    img_mask,
                    self.temperature,
                    self.beta,
                )
                kl_loss = (
                    self.kl_loss(global_t2v_logits, local_t2v_logits)
                    + self.kl_loss(global_v2t_logits, local_v2t_logits)
                ) / 2
                centrality_loss = self.compute_centrality_loss(
                    text_feat,
                    img_feat,
                    global_text_feat,
                    global_img_feat,
                    local_t2v_logits,
                    local_v2t_logits,
                    self.centrality_scale,
                    self.logit_scale,
                )
                neighbor_loss = self.compute_neighbor_loss(
                    text_feat,
                    img_feat,
                    text_mask,
                    img_mask,
                    mb_feat_t,
                    mb_feat_v,
                    mb_mask_t,
                    mb_mask_v,
                    local_t2v_logits,
                    local_v2t_logits,
                    self.num_neighbors,
                    self.temperature,
                )

                total_loss = (
                    centrality_loss
                    + (uniform_loss * self.uniform_weight)
                    + (neighbor_loss * self.neighbor_weight)
                    + (kl_loss * self.kl_weight)
                )

                # F_I = torch.autograd.Variable(img_hash)
                # F_T = torch.autograd.Variable(text_hash)
                # F_I1, code_I = img_hash, torch.tanh(img_hash)
                # F_T1, code_T = text_hash, torch.tanh(text_hash)
                # J = torch.cat((F_I1, F_T1), 1)
                # code_J = self.CodeNet_J(J)
                # view1_predict, view2_predict, _ = self.gcn_L(F_I1, F_T1)

                # S_I = euclidean_dist(F_I, F_I)
                # S_I = torch.exp(-S_I / 4)
                # S_T = cosine_dist(F_T, F_T)

                # F_BI, B_GI = self.gcn_I(F_I1, S_I)
                # F_BT, B_GT = self.gcn_T(F_T1, S_T)

                # B_I = F.normalize(code_I)
                # B_T = F.normalize(code_T)
                # B_J = F.normalize(code_J)
                # B_GI = F.normalize(B_GI)
                # B_GT = F.normalize(B_GT)

                # BI_BI = B_I.mm(B_I.t())
                # BT_BT = B_T.mm(B_T.t())
                # BI_BJ = B_I.mm(B_J.t())
                # BT_BJ = B_T.mm(B_J.t())
                # BJ_BJ = B_J.mm(B_J.t())
                # B_BGI = B_GI.mm(B_GI.t())
                # B_BGT = B_GT.mm(B_GT.t())

                # loss1 = (
                #     F.mse_loss(BI_BI, S_I)
                #     + F.mse_loss(BT_BT, S_T)
                #     + F.mse_loss(BJ_BJ, S_I)
                #     + F.mse_loss(BJ_BJ, S_T)
                # )
                # loss2 = F.mse_loss(B_BGI, S_I) + F.mse_loss(B_BGT, S_T)
                # loss3 = F.mse_loss(BI_BJ, S_I) + F.mse_loss(BT_BJ, S_T)
                # loss4 = F.mse_loss(B_I, B_GI) + F.mse_loss(B_T, B_GT)
                # loss5 = F.mse_loss(B_I, B_J) + F.mse_loss(B_T, B_J)
                # loss6 = calc_loss(view1_predict, view2_predict, label, label)
                loss = self.dsph(img_hash, text_hash, label)
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_dsph.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.optimizer_dsph.step()
            self.valid(i, loss.item())

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

        # img_buffer = torch.empty(length, cfg["num_bit"], dtype=torch.float).to(
        #     cfg["device"]
        # )
        # text_buffer = torch.empty(length, cfg["num_bit"], dtype=torch.float).to(
        #     cfg["device"]
        # )
        # for image, text, padding_mask, label, index in tqdm(data_loader):
        #     image, text, label = (
        #         image.to(cfg["device"], non_blocking=True),
        #         text.to(cfg["device"], non_blocking=True),
        #         label.to(cfg["device"], non_blocking=True).float(),
        #     )
        #     index = index.numpy()
        #     # hash_img, hash_text, _, _, _ = self.model(image, text, padding_mask, label)
        #     out_dict = self.model(image, text, padding_mask, label)
        #     hash_img = out_dict["image_hash_64"]
        #     hash_text = out_dict["text_hash_64"]
        #     hash_img = torch.sign(hash_img.detach())
        #     hash_text = torch.sign(hash_text.detach())
        #     img_buffer[index, :] = hash_img.data
        #     text_buffer[index, :] = hash_text.data
        # return img_buffer, text_buffer

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
            (0, 1, cfg["out_dim"]), dtype=torch.float, device=device
        )
        self.mb_feat_v = torch.empty(
            (0, 1, cfg["out_dim"]), dtype=torch.float, device=device
        )
        self.mb_mask_t = torch.empty((0, 1), dtype=torch.float, device=device)
        self.mb_mask_v = torch.empty((0, 1), dtype=torch.float, device=device)
        self.mb_batch = 0

    def reshape_feat_gener_mask(bs, text_vec, img_vec):
        """
        输入：CLIP 输出的 (bs, 512) 句子/图片向量
        返回：单 Token 序列， mask 全 1
        """
        device = text_vec.device
        text_feat = text_vec.unsqueeze(1)  # (bs, 1, 512)
        img_feat = img_vec.unsqueeze(1)  # (bs, 1, 512)
        text_mask = torch.ones(bs, 1, dtype=torch.long, device=device)
        img_mask = torch.ones(bs, 1, dtype=torch.long, device=device)
        return text_feat, img_feat, text_mask, img_mask

    def merge_global_features(bs, text_vec, img_vec):
        """
        输入：CLIP 输出的 (bs, 512) 句子/图片向量
        返回：单 Token 序列， mask 全 1
        """
        device = cfg["device"]
        text_feat = text_vec.unsqueeze(1)  # (bs, 1, 512)
        img_feat = img_vec.unsqueeze(1)  # (bs, 1, 512)
        text_mask = torch.ones(bs, 1, dtype=torch.long, device=device)
        img_mask = torch.ones(bs, 1, dtype=torch.long, device=device)
        return text_feat, img_feat, text_mask, img_mask

    def compute_centrality_loss(
        self,
        text_feat,
        video_feat,
        global_text_feat,
        global_video_feat,
        local_t2v_logits,
        local_v2t_logits,
        centrality_scale,
        logit_scale,
    ):
        # Calculate local centrality weights
        local_text_weights, local_video_weights = self.compute_centrality_weights(
            text_feat, video_feat, global_text_feat, global_video_feat, centrality_scale
        )

        # Apply centrality weighting to entity-level logits
        centrality_loss_t2v = self.centrality_weighting_loss(
            local_t2v_logits * logit_scale, local_text_weights
        )
        centrality_loss_v2t = self.centrality_weighting_loss(
            local_v2t_logits * logit_scale, local_video_weights
        )

        # Average bidirectional losses
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
          text_feat        : (B,1,512)  ← 来自 clip_to_merge
          image_feat       : (B,1,512)
          global_text_feat : (B,1,512)  ← 可再经过一层 attention 后的全局向量
          global_image_feat: (B,1,512)
          centrality_scale : float
        返回：
          text_weights  : (B,)
          image_weights : (B,)
        """
        # 直接 squeeze 掉单 token 维度
        text_local = text_feat.squeeze(1)  # (B,512)
        image_local = image_feat.squeeze(1)  # (B,512)
        text_global = global_text_feat.squeeze(1)  # (B,512)
        image_global = global_image_feat.squeeze(1)  # (B,512)

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
        video_feat,
        text_mask,
        video_mask,
        mb_feat_t,
        mb_feat_v,
        mb_mask_t,
        mb_mask_v,
        local_t2v_logits,
        local_v2t_logits,
        num_neighbors,
        temperature,
    ):
        # Calculate memory bank logits
        memory_bank_t2v_logits, _ = self.local_level(
            text_feat, mb_feat_v, text_mask, mb_mask_v
        )
        _, memory_bank_v2t_logits = self.local_level(
            mb_feat_t, video_feat, mb_mask_t, video_mask
        )

        # Apply neighbor adjusting loss
        neighbor_loss_t2v = self.neighbor_adjusting_loss(
            local_t2v_logits, memory_bank_v2t_logits, num_neighbors, temperature
        )
        neighbor_loss_v2t = self.neighbor_adjusting_loss(
            local_v2t_logits, memory_bank_t2v_logits, num_neighbors, temperature
        )

        # Average bidirectional losses
        return (neighbor_loss_t2v + neighbor_loss_v2t) / 2

    def compute_uniform_loss(
        self, text_feat, video_feat, text_mask, video_mask, temperature, beta
    ):
        # Get global features through hierarchical token merging
        global_text_feat, global_video_feat = self.merge_global_features(
            text_feat, video_feat, text_mask, video_mask
        )

        # Calculate similarity logits at global level
        global_t2v_logits, global_v2t_logits = self.global_level(
            global_text_feat, global_video_feat
        )

        # Calculate uniform regularization loss
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
            global_video_feat,
            global_t2v_logits,
            global_v2t_logits,
        )


if __name__ == "__main__":
    WYL_Trainer().train()
