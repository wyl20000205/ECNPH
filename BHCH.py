import numpy as np
import torch
import torch.nn as nn
from config import cfg
from open_clip import create_model_from_pretrained, create_model_and_transforms
from torch.nn.functional import normalize
from mamba_ssm import Mamba, Mamba2
from model_dsph import clip
from einops import rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer, init
import torch.nn.functional as F
from model import open_clip
from torch.nn import Parameter


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + normalize(x, p=2, dim=-1)


class MLPLayer(nn.Module):
    """
    LND - LND or ND - ND
    """

    # 64 -> 128 -> 128 auxiliary
    def __init__(self, dim_list, dropout=0):
        super().__init__()

        self.activation_layer = nn.ReLU()
        self.mlp = nn.Sequential()

        for i in range(len(dim_list) - 2):
            _in = dim_list[i]
            _out = dim_list[i + 1]
            self.mlp.add_module(f"linear_{i}", nn.Linear(_in, _out))
            self.mlp.add_module(f"activate_{i}", self.activation_layer)
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(p=dropout))
        self.mlp.add_module(f"linear_final", nn.Linear(dim_list[-2], dim_list[-1]))

    def forward(self, x):
        return self.mlp(x)


class HashingEncoder(nn.Module):
    def __init__(self, org_dim, k_bits):
        super().__init__()
        self.fc = nn.Linear(org_dim, k_bits)
        self.drop_out = nn.Dropout(p=cfg["dropout"])

    def forward(self, x):
        x = self.drop_out(self.fc(x))
        return torch.tanh(x)


class HashingDecoder(nn.Module):
    """
    hashing decoder, MLP & tach.
    """

    def __init__(self, org_bit_dim, recon_bit_dim):
        super().__init__()
        self.mlp = MLPLayer(dim_list=[org_bit_dim, recon_bit_dim, recon_bit_dim])
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        return torch.tanh(self.mlp(x))


class ExtraLinear(nn.Module):
    def __init__(self, inputDim=512, outputDim=cfg["num_bit"]):
        super(ExtraLinear, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)

    def forward(self, x):
        return self.fc(x)


def log_scaled_softmax(scores, s=0.5):
    d = scores.shape[-1]  # 默认取最后一个维度
    n = np.arange(1, scores.shape[-1] + 1)
    log_weights = s * np.log(n)
    scaled_scores = (log_weights * scores) / np.sqrt(d)
    e_x = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    return (
        torch.from_numpy(e_x / e_x.sum(axis=-1, keepdims=True))
        .to(cfg["device"])
        .float()
    )


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, image_embed, text_embed):
        B, T = image_embed.shape

        q = image_embed.unsqueeze(0)  # 1 50 768
        k = text_embed.unsqueeze(0).permute(0, 2, 1)  # 1 768 50
        att_map = torch.bmm(q, k)
        att_map = torch.softmax(att_map, dim=-1)

        v = text_embed.unsqueeze(0)
        out = torch.bmm(att_map, v)
        out = out.squeeze(0)
        return torch.cat((self.alpha * out, image_embed), dim=1)


class MambaEncoder(nn.Module):
    def __init__(self, d_model=cfg["out_dim"] * 2, d_state=16, d_conv=4, expand=2):
        super(MambaEncoder, self).__init__()
        self.cam = CAM()
        self.mamba = Mamba(
            d_model=d_model,  # Model dimension d_model # 图片和文字输出维度 * 2
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    # def feature_enhance(self, image_embed, text_embed):
    #     i1 = torch.sum(image_embed, dim=1)
    #     t1 = torch.sum(text_embed, dim=1)
    #     mi = i1.unsqueeze(1) @ i1.unsqueeze(0)
    #     mt = t1.unsqueeze(1) @ t1.unsqueeze(0)
    #     similar_matrix = mi - mt
    #     similar_matrix = (
    #         (1 - torch.tanh(similar_matrix) ** 2)
    #         * torch.sigmoid(similar_matrix)
    #         * (1 - torch.sigmoid(similar_matrix))
    #     )
    #     feature_a = similar_matrix @ image_embed
    #     feature_b = similar_matrix @ text_embed
    #     feature_c = torch.cat((feature_a, feature_b), dim=1)
    #     return 0.1 * feature_c

    def forward(self, image_embed, text_embed):
        tokens = torch.concat((image_embed, text_embed), dim=1)  # 1 bs 1024
        tokens = tokens.unsqueeze(0)  # 1 bs 1024
        result = self.mamba(tokens).squeeze()  # bs 1024
        # self.feature_enhance(image_embed, text_embed)
        result = result + self.cam(image_embed, text_embed)  # bs 512 + bs 512 = bs 512
        result = normalize(result, p=2, dim=1)
        return result.chunk(2, dim=1)
        return (
            result[:, : cfg["out_dim"]],
            result[:, cfg["out_dim"] :],
        )


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.nin = nn.Linear(dim, dim)
        self.nin2 = nn.Linear(dim, dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.norm2 = GRN(dim=dim)
        self.act2 = nn.SiLU()
        self.act3 = nn.SiLU()

        # self.norm = nn.LayerNorm(dim)
        self.norm = GRN(dim=dim)
        self.act = nn.SiLU()
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.temp = nn.Parameter(torch.randn(1).abs(), requires_grad=True)
        self.weight_gate = nn.Parameter(torch.randn(1).abs(), requires_grad=True)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        B, N, C = x.shape
        x = self.nin(x)
        x = self.norm(x)
        x = self.act(x)
        act_x = x
        assert C == self.dim
        n_tokens = x.shape[1:-1].numel()
        img_dims = x.shape[1:-1]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_flip_l = torch.flip(x_flat, dims=[2])
        x_flip_c = torch.flip(x_flat, dims=[1])
        x_flip_lc = torch.flip(x_flat, dims=[1, 2])
        x_ori = self.mamba(x_flat)
        x_mamba_l = self.mamba(x_flip_l)
        x_mamba_c = self.mamba(x_flip_c)
        x_mamba_lc = self.mamba(x_flip_lc)
        x_ori_l = torch.flip(x_mamba_l, dims=[2])
        x_ori_c = torch.flip(x_mamba_c, dims=[1])
        x_ori_lc = torch.flip(x_mamba_lc, dims=[1, 2])
        x_mamba = (x_ori + x_ori_l + x_ori_c + x_ori_lc) * self.temp

        out = x_mamba.transpose(-1, -2).reshape(B, *img_dims, C)
        cos_sim = F.cosine_similarity(out, act_x, dim=-1)
        weight = torch.sigmoid(cos_sim.mean(0)).unsqueeze(-1)

        out = self.weight_gate * weight * out + (1 - weight) * act_x
        out = self.nin2(out)
        out = self.norm2(out)
        out = self.act2(out)
        return out


class FusionTransMamba(nn.Module):
    def __init__(self, num_layers=1, hidden_size=cfg["out_dim"] * 2, nhead=4):
        super(FusionTransMamba, self).__init__()
        self.d_model = hidden_size
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, batch_first=False
        )
        self.transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.sigal_d = cfg["out_dim"]
        self.inproj = nn.Linear(cfg["out_dim"], cfg["out_dim"])
        self.outproj = nn.Linear(cfg["out_dim"], cfg["out_dim"])
        self.mamba = MambaLayer(dim=cfg["out_dim"], d_state=16, d_conv=4, expand=2)
        # self.grn1 = nn.LayerNorm(self.sigal_d)
        # self.grn2 = nn.LayerNorm(self.d_model)
        self.grn1 = GRN(dim=self.sigal_d)
        self.grn2 = GRN(dim=self.d_model)

    def weight_init(self):
        self.inproj.apply(self.kaiming_init)
        self.outproj.apply(self.kaiming_init)

    def kaiming_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif classname.find("Linear") != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif classname.find("Norm") != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    def forward(self, img_cls, txt_eos):
        short_img_cls = self.inproj(img_cls)
        short_txt_eos = self.inproj(txt_eos)
        mamba_att = self.mamba(
            torch.concat((img_cls.unsqueeze(1), txt_eos.unsqueeze(1)), dim=1)
        )
        img_cls, txt_eos = torch.chunk(mamba_att, chunks=2, dim=0)
        img_cls = self.outproj(self.grn1(img_cls).squeeze())
        txt_eos = self.outproj(self.grn1(txt_eos).squeeze())
        img_cls = 0.5 * img_cls + 0.5 * short_img_cls
        txt_eos = 0.5 * txt_eos + 0.5 * short_txt_eos
        res_temp_cls = torch.concat((img_cls, txt_eos), dim=-1)
        res_temp_cls = self.grn2(res_temp_cls)
        encoder_X = self.transformer(res_temp_cls)
        encoder_X_r = encoder_X.reshape(-1, self.d_model)
        encoder_X_r = normalize(encoder_X_r, p=2, dim=-1)
        img_cls, txt_eos = (
            encoder_X_r[:, : self.sigal_d],
            encoder_X_r[:, self.sigal_d :],
        )
        return img_cls, txt_eos


class VLPromptLearner(nn.Module):
    def __init__(self, clip_model, n_cls, maxWords, device):
        super().__init__()
        self.device = device
        self.maxWords = maxWords
        self.tokenizer = open_clip.SimpleTokenizer()
        self.ctx_init = "This is an image containing"
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
        }

        with torch.no_grad():
            self.token_embedding = clip_model.token_embedding.to(device)
            for param in self.token_embedding.parameters():
                param.requires_grad = False
        self.ctx_dim = cfg["out_dim"]
        # 预计算特殊token
        special_tokens = [
            self.SPECIAL_TOKEN["CLS_TOKEN"],
            self.SPECIAL_TOKEN["SEP_TOKEN"],
        ]
        self.special_token_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(special_tokens),
            device=device,
            dtype=torch.long,
        )

        ctx_vectors = torch.empty(self.maxWords, self.ctx_dim, device=device)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        self.register_buffer(
            "padding_zeros", torch.zeros(self.maxWords, dtype=torch.long, device=device)
        )
        self.batch_buffer = None

        self._prompt_cache = {}

    def replace_underscore(self, name_list):
        cache_key = tuple(name_list)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        # 快速字符串处理
        if len(name_list) > 1:
            joined_names = ", ".join(n.replace("_", " ") for n in name_list[:-1])
            joined_names += f" and {name_list[-1].replace('_', ' ')}"
        else:
            joined_names = name_list[0].replace("_", " ")

        # 处理tokens
        tokens = self.tokenizer.tokenize(f"{self.ctx_init} {joined_names}.")
        token_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(tokens[: self.maxWords - 2]),
            device=self.device,
            dtype=torch.long,
        )

        prompt_ids = self.padding_zeros.clone()
        prompt_ids[0] = self.special_token_ids[0]  # CLS
        length = token_ids.size(0)
        prompt_ids[1 : length + 1].copy_(token_ids)
        prompt_ids[length + 1] = self.special_token_ids[1]  # SEP

        if len(self._prompt_cache) < 1000:
            self._prompt_cache[cache_key] = prompt_ids
        return prompt_ids

    def clear_cache(self):
        self._prompt_cache.clear()
        torch.cuda.empty_cache()

    @torch.amp.autocast(device_type="cuda")
    def forward(self, classnames):
        batch_size = len(classnames)

        if self.batch_buffer is None or self.batch_buffer.size(0) != batch_size:
            self.batch_buffer = torch.empty(
                (batch_size, self.maxWords), dtype=torch.long, device=self.device
            )

        for i, name_list in enumerate(classnames):
            self.batch_buffer[i] = self.replace_underscore(name_list)

        prompts = self.batch_buffer[:batch_size]
        embedding = self.token_embedding(prompts)
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)

        return embedding + ctx


class BHCH(nn.Module):
    def __init__(self):
        super(BHCH, self).__init__()
        self.grn = GRN(cfg["out_dim"] * 2)
        embedDim, self.clip = self._load_clip()  # style 1: load clip
        self.class_name_list = cfg["class_name_list"]
        # self.clip, _, _ = create_model_and_transforms( # style 2:load clip
        #     cfg["model_name"], device=cfg["device"]
        # )

        # self.clip, _, _ = open_clip.create_model_and_transforms(
        # "ViT-B-32-quickgelu", pretrained="metaclip_fullcc"
        # )  # style 3:load clip
        embedDim = cfg["out_dim"]
        self.hash_encoders = nn.ModuleList(
            HashingEncoder(org_dim=cfg["out_dim"], k_bits=one)
            for one in cfg["list_bit"]
        )
        self.image_pre = ExtraLinear(inputDim=embedDim).to(cfg["device"]).float()
        self.text_pre = ExtraLinear(inputDim=embedDim).to(cfg["device"]).float()
        self.label = (
            ExtraLinear(inputDim=cfg["num_class"], outputDim=cfg["num_bit"])
            .to(cfg["device"])
            .float()
        )
        self.FuseTrans = MambaEncoder(cfg["out_dim"] * 2)
        self.image_silu = nn.SiLU()
        self.text_silu = nn.SiLU()
        self.prompt_learner = VLPromptLearner(
            clip_model=self.clip,
            n_cls=1,
            device=cfg["device"],
            maxWords=cfg["max_words"],
        )
        self.FuseMamba = FusionTransMamba(
            num_layers=1, hidden_size=cfg["out_dim"] * 2, nhead=4
        )

    def _encode_image(self, image):
        image_embed_1 = self.clip.encode_image(image)  # bs 512
        image_pre = self.image_pre(image_embed_1)
        return image_embed_1, image_pre  # bs 32

    def _encode_text(self, text):
        text_embed_1 = self.clip.encode_text(text)  # bs 512
        text_pre = self.text_pre(text_embed_1)
        return text_embed_1, text_pre

    def _encode_label(self, label):
        text_embed_1 = self.label(label)
        return text_embed_1

    def _load_clip(self, clipPath=cfg["name_clip"]):
        model, _ = clip.load(clipPath, device=cfg["device"])
        model.float()
        return cfg["out_dim"], model

    def forward(self, image, text, label):
        image_features, image_pre = self._encode_image(image)
        text_features, text_pre = self._encode_text(text)
        label_features = self._encode_label(label)
        image_features, text_features = normalize(
            image_features, p=2, dim=1
        ), normalize(text_features, p=2, dim=1)
        return image_features, text_features, image_pre, text_pre


import math
from scipy.io import loadmat


def gen_A(num_classes, t, adj_file):
    result = loadmat(adj_file)
    _adj = result["adj"]
    _nums = result["nums"]
    _nums = _nums[:, None]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj.squeeze()
    _adj = _adj * 0.5
    _adj = _adj + np.identity(_adj.shape[0], float)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(axis=1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class JNet(nn.Module):  # 特征融合模块
    def __init__(self, code_len):
        super(JNet, self).__init__()
        self.fc_encode = nn.Linear(cfg["num_bit"] * 2, code_len).to(cfg["device"])
        self.alpha = 1.0

    def forward(self, x):
        code = torch.tanh(self.alpha * self.fc_encode(x))
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCNLT(nn.Module):
    def __init__(self, code_len):
        super(GCNLT, self).__init__()
        self.gcn1 = GCN(cfg["num_bit"], cfg["num_bit"]).to("cuda:0")
        self.gcn2 = GCN(cfg["num_bit"], cfg["num_bit"]).to("cuda:0")
        self.gcn3 = GCN(cfg["num_bit"], cfg["num_bit"]).to("cuda:0")
        self.relu = nn.LeakyReLU(0.2).to("cuda:0")
        self.hypo = nn.Linear(3 * cfg["num_bit"], cfg["num_bit"]).to("cuda:0")
        self.fc_encode = nn.Linear(cfg["num_bit"], code_len).to("cuda:0")
        self.alpha = 1.0

    def forward(self, input, adj):
        layers = []

        x = self.gcn1(input, adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn2(x, adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn3(x, adj)
        x = self.relu(x)
        layers.append(x)
        x = torch.cat(layers, -1)
        x = self.hypo(x)
        code = torch.tanh(self.alpha * self.fc_encode(x))

        return x, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNLI(nn.Module):
    def __init__(self, code_len=cfg["num_bit"]):

        super(GCNLI, self).__init__()

        # self.num_classes = num_classes
        self.gcn1 = GCN(cfg["num_bit"], cfg["num_bit"]).to(cfg["device"])
        self.gcn2 = GCN(cfg["num_bit"], cfg["num_bit"]).to(cfg["device"])
        self.gcn3 = GCN(cfg["num_bit"], cfg["num_bit"]).to(cfg["device"])
        self.relu = nn.LeakyReLU(0.2).to(cfg["device"])
        self.hypo = nn.Linear(3 * cfg["num_bit"], cfg["num_bit"]).to(cfg["device"])
        self.fc_encode = nn.Linear(cfg["num_bit"], code_len).to(cfg["device"])
        self.alpha = 1.0

    def forward(self, input, adj):
        layers = []

        x = self.gcn1(input, adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn2(x, adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn3(x, adj)
        x = self.relu(x)
        layers.append(x)
        x = torch.cat(layers, -1)  # 将多个特征x按最后一个维度进行拼接
        x = self.hypo(x)  # 将拼接后的特征x通过线性层hypo进行映射
        code = torch.tanh(self.alpha * self.fc_encode(x))

        return x, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNL(nn.Module):
    def __init__(
        self,
        minus_one_dim=cfg["num_bit"],
        num_classes=10,
        in_channel=300,
        t=0,
        adj_file="/home/tus/Desktop/wyl/MLGCH/adj.mat",
        inp="/home/tus/Desktop/wyl/MLGCH/mirflickr-inp-glove6B.mat",
    ):

        super(GCNL, self).__init__()
        inp = loadmat(inp)["inp"]
        inp = torch.FloatTensor(inp)
        self.gcn1 = GCN(in_channel, minus_one_dim)
        self.gcn2 = GCN(minus_one_dim, minus_one_dim)
        self.gcn3 = GCN(minus_one_dim, minus_one_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.hypo = nn.Linear(3 * minus_one_dim, minus_one_dim)

        _adj = torch.FloatTensor(gen_A(num_classes, t, adj_file))

        self.adj = Parameter(gen_adj(_adj), requires_grad=False)

        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))

    def forward(self, feature_img, feature_text):
        view1_feature = feature_img
        view2_feature = feature_text

        layers = []

        x = self.gcn1(self.inp, self.adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn2(x, self.adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn3(x, self.adj)
        x = self.relu(x)
        layers.append(x)
        x = torch.cat(layers, -1)
        x = self.hypo(x)

        norm_img = (
            torch.norm(view1_feature, dim=1)[:, None]
            * torch.norm(x.cuda(), dim=1)[None, :]
            + 1e-6
        )
        norm_txt = (
            torch.norm(view2_feature, dim=1)[:, None]
            * torch.norm(x.cuda(), dim=1)[None, :]
            + 1e-6
        )
        x = x.transpose(0, 1)
        y_img = torch.matmul(view1_feature, x.cuda())
        y_text = torch.matmul(view2_feature, x.cuda())
        y_img = y_img / norm_img
        y_text = y_text / norm_txt

        return y_img, y_text, x.transpose(0, 1)
