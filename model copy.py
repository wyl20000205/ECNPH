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
    def __init__(self, inputDim=512, outputDim=cfg["num_class"]):
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
    def __init__(self, d_model=1024, d_state=16, d_conv=4, expand=2):
        super(MambaEncoder, self).__init__()
        self.cam = CAM()
        self.mamba = Mamba(
            d_model=d_model,  # Model dimension d_model # 图片和文字输出维度 * 2
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    def feature_enhance(self, image_embed, text_embed):
        i1 = torch.sum(image_embed, dim=1)
        t1 = torch.sum(text_embed, dim=1)
        mi = i1.unsqueeze(1) @ i1.unsqueeze(0)
        mt = t1.unsqueeze(1) @ t1.unsqueeze(0)
        similar_matrix = mi - mt
        similar_matrix = (
            (1 - torch.tanh(similar_matrix) ** 2)
            * torch.sigmoid(similar_matrix)
            * (1 - torch.sigmoid(similar_matrix))
        )
        feature_a = similar_matrix @ image_embed
        feature_b = similar_matrix @ text_embed
        feature_c = torch.cat((feature_a, feature_b), dim=1)
        return 0.1 * feature_c

    def forward(self, image_embed, text_embed):
        tokens = torch.concat((image_embed, text_embed), dim=1)
        tokens = tokens.unsqueeze(0)
        result = self.mamba(tokens).squeeze()  # [50, 1536]
        # self.feature_enhance(image_embed, text_embed)
        result = result + self.cam(image_embed, text_embed)
        result = normalize(result, p=2, dim=1)
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
            expand=expand  # Block expansion factor
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
    def __init__(self,  num_layers=1, hidden_size=cfg["out_dim"]*2, nhead=4):
        super(FusionTransMamba, self).__init__()
        self.d_model = hidden_size
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, batch_first=False)
        self.transformer = TransformerEncoder(encoder_layer= encoder_layer, num_layers= num_layers, enable_nested_tensor=False)
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
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif classname.find('Norm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    def forward(self, img_cls, txt_eos):
        short_img_cls = self.inproj(img_cls)
        short_txt_eos = self.inproj(txt_eos)
        mamba_att = self.mamba(torch.concat((img_cls.unsqueeze(1), txt_eos.unsqueeze(1)), dim = 1))
        img_cls, txt_eos = torch.chunk(mamba_att, chunks=2, dim=0)
        img_cls = self.outproj(self.grn1(img_cls).squeeze())
        txt_eos = self.outproj(self.grn1(txt_eos).squeeze())
        img_cls = 0.5 * img_cls + 0.5 * short_img_cls
        txt_eos = 0.5 * txt_eos + 0.5 * short_txt_eos
        res_temp_cls = torch.concat((img_cls, txt_eos), dim = -1)
        res_temp_cls = self.grn2(res_temp_cls)
        encoder_X = self.transformer(res_temp_cls)
        encoder_X_r = encoder_X.reshape( -1,self.d_model)
        encoder_X_r = normalize(encoder_X_r, p =2 ,dim =-1)
        img_cls, txt_eos = encoder_X_r[:,:self.sigal_d], encoder_X_r[:,self.sigal_d:]
        return img_cls, txt_eos
    

from model_dsph.simple_tokenizer import SimpleTokenizer as Tokenizer

class VLPromptLearner(nn.Module):
    def __init__(self, clip_model, n_cls, maxWords, device):
        super().__init__()
        self.device = device
        self.maxWords = maxWords
        self.tokenizer = Tokenizer()
        self.ctx_init = "This is an image containing"
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>"}
        
        # 保存并冻结embedding层
        # print(clip_model.text.token_embedding) #TextTransformer #token_embedding
        with torch.no_grad():
            self.token_embedding = clip_model.text.token_embedding.to(device)
            for param in self.token_embedding.parameters():
                param.requires_grad = False
        # print(type(clip_model.text.token_embedding))
        self.ctx_dim = cfg["out_dim"]
        
        # 预计算特殊token
        special_tokens = [self.SPECIAL_TOKEN["CLS_TOKEN"], self.SPECIAL_TOKEN["SEP_TOKEN"]]
        self.special_token_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(special_tokens),
            device=device, dtype=torch.long
        )
        
        ctx_vectors = torch.empty(self.maxWords, self.ctx_dim, device=device)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        self.register_buffer('padding_zeros', torch.zeros(self.maxWords, dtype=torch.long, device=device))
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
            self.tokenizer.convert_tokens_to_ids(tokens[:self.maxWords-2]),
            device=self.device, dtype=torch.long
        )
        
        # 使用预分配buffer
        prompt_ids = self.padding_zeros.clone()
        prompt_ids[0] = self.special_token_ids[0]  # CLS
        length = token_ids.size(0)
        prompt_ids[1:length+1].copy_(token_ids)
        prompt_ids[length+1] = self.special_token_ids[1]  # SEP
        
        if len(self._prompt_cache) < 1000:
            self._prompt_cache[cache_key] = prompt_ids
        return prompt_ids

    def clear_cache(self):
        self._prompt_cache.clear()
        torch.cuda.empty_cache()
        
    @torch.amp.autocast(device_type='cuda')
    def forward(self, classnames):
        batch_size = len(classnames)
        
        if self.batch_buffer is None or self.batch_buffer.size(0) != batch_size:
            self.batch_buffer = torch.empty(
                (batch_size, self.maxWords),
                dtype=torch.long,
                device=self.device
            )
        
        for i, name_list in enumerate(classnames):
            self.batch_buffer[i] = self.replace_underscore(name_list)
        
        prompts = self.batch_buffer[:batch_size]
        embedding = self.token_embedding(prompts)
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)
        
        return embedding + ctx

        
class DPDH_Encoder(nn.Module):
    def __init__(self):
        super(DPDH_Encoder, self).__init__()
        self.grn = GRN(768*2)
        # embedDim, self.clip = self._load_clip()  # style 1: load clip
        self.class_name_list = cfg["class_name_list"]
        self.clip, _, _ = create_model_and_transforms(
            cfg["model_name"], device=cfg["device"]
        )
        embedDim = 768
        self.hash_encoders = nn.ModuleList(
            HashingEncoder(org_dim=768, k_bits=one) for one in cfg["list_bit"]
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
        # print(self.clip)
        self.prompt_learner = VLPromptLearner(clip_model=self.clip, n_cls=1, device=cfg["device"], maxWords=cfg["max_words"])
        
    def _encode_image(self, image):
        image_embed_1 = self.clip.encode_image(image)  # bs 768
        image_pre = self.image_pre(image_embed_1)
        return image_embed_1, image_pre  # bs 32

    def _encode_text(self, text,text_prompt = None):
        text_embed_1 = self.clip.encode_text(text)  # bs 768
        text_pre = self.text_pre(text_embed_1)
        return text_embed_1, text_pre

    def _encode_label(self, label):
        text_embed_1 = self.label(label)
        return text_embed_1

    def _load_clip(self, clipPath=cfg["name_clip"]):
        model, _ = clip.load(clipPath, device=cfg["device"])
        model.float()
        return cfg["out_dim"], model

    def forward(self, image, text, key_padding_mask, label):
        out_dict = {}
        # short_img_cls = img_cls
        # short_txt_cls = txt_eos
        # short_prompt_eos = prompt_eos
        B = image.shape[0]
        indices = [torch.where(label[i] == 1)[0] for i in range(B)]
        class_names_prompt = [[self.class_name_list[j] for j in indices[i]] for i in range(B)]
        text_prompt = self.prompt_learner(class_names_prompt)
        # print(text_prompt.shape) #30, 64, 768
        
        image_features, image_pre = self._encode_image(image)
        text_features, text_pre = self._encode_text(text,text_prompt)
        
        label_features = self._encode_label(label)
        
        image_features, text_features = normalize(image_features, p=2, dim=1), normalize(text_features, p=2, dim=1)
        image_features, text_features = self.grn(torch.cat((image_features, text_features), dim=1)).chunk(2, dim=1)
        image_features = self.image_silu(image_features)
        text_features = self.text_silu(text_features)
        image_features, text_features = self.FuseTrans(
            image_features, text_features
        )  # bs 768
        image_features = self.image_silu(image_features)
        text_features = self.text_silu(text_features)
        
        # img_cls, txt_eos = self.FuseMamba(img_cls, txt_eos) #[50, 512]) torch.Size([50, 512]
        # img_cls, prompt_eos = self.FuseMamba(img_cls, prompt_eos) #[50, 512]) torch.Size([50, 512]
        
        # img_cls = 0.5 * short_img_cls + 0.5 * img_cls
        # txt_eos = 0.5 * short_txt_cls + 0.5 * txt_eos
        # prompt_eos = 0.5 * short_prompt_eos + 0.5 * prompt_eos
        
        
        
        for i, one in enumerate(cfg["list_bit"]):
            image_hash = self.hash_encoders[i](image_features)
            text_hash = self.hash_encoders[i](text_features)
            out_dict[f"image_hash_{one}"] = image_hash
            out_dict[f"text_hash_{one}"] = text_hash
        out_dict["image_pre"] = image_pre
        out_dict["text_pre"] = text_pre
        out_dict["label_features"] = label_features
        # image_features = normalize(image_features, p=2, dim=1)
        # text_features = normalize(text_features, p=2, dim=1)

        return out_dict





# if __name__ == "__main__":
    # t_a = Tiny_attention()
    # x = torch.randn(1,50,768)
    # x = t_a(x)
    # print(x.shape)
    # image_embed = torch.randn(50, 768)
    # text_embed = torch.randn(50, 768)
    # cam = CAM()
    # out = cam(image_embed, text_embed)
    # print(out.shape)
