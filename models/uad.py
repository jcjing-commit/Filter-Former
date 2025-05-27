import torch
import torch.nn as nn
import torch.nn.functional as F
import math



from torch import nn
from torchvision.datasets import ImageFolder

from mamba_ssm import Mamba

# import torch.nn.functional as F
# from flash_attn import flash_attn_qkvpacked
# from torch.utils.checkpoint import checkpoint


def get_autoencoder_org(out_channels=384, in_channel=3, side=64):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=side, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )


def get_autoencoder(out_channels=384, in_channel=3, side=64):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=8),
        # decoder
        nn.Upsample(size=7, mode='bilinear'),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=64, mode='bilinear'),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        # nn.Upsample(size=128, mode='bilinear'),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        # nn.Upsample(size=128, mode='bilinear'),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=side, mode='bilinear'),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )


def get_autoencoder_v2(out_channels=384, in_channel=768, side=64):
    return nn.Sequential(
        # encoder
        # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2,
        #           padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2,
        #           padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2,
        #           padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2,
        #           padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2,
        #           padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=8),
        # decoder
        nn.Upsample(size=side//4, mode='bilinear'),
        nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=side//2, mode='bilinear'),
        nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=side, mode='bilinear'),
        nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=side*2, mode='bilinear'),
        nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=side*2, mode='bilinear'),
        nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=side, mode='bilinear'),
        nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=side, mode='bilinear'),
        nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )




class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, side, scale_factor):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(size=side // scale_factor, mode='bilinear')
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

def get_autoencoder_v3(out_channels=384, in_channel=768, side=64):
    return nn.Sequential(
        # Decoder
        DecoderBlock(in_channel, in_channel//2, side, 4),
        DecoderBlock(in_channel//2, in_channel//4, side, 2),
        DecoderBlock(in_channel//4, in_channel//4, side, 1),
        DecoderBlock(in_channel//4, in_channel//2, side, 2),
        DecoderBlock(in_channel//2, in_channel//2, side, 2),
        DecoderBlock(in_channel//2, in_channel, side, 1),
        DecoderBlock(in_channel, in_channel, side, 1),
        nn.Upsample(size=side, mode='bilinear'),
        nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )


class MultiHeadSelfAttention_v1(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, ff_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by num_heads"

        # QKV生成层
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 替换原线性层为MLP
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # 生成QKV并拆分多头
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 计算注意力权重
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权和与合并多头
        out = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        # 通过MLP输出
        return self.out_proj(out) + x


class MultiHeadSelfAttention_v1_org(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, ff_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 使用 PyTorch 的 MultiheadAttention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # MLP
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),  # 使用更节省显存的激活函数
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        # batch_size, seq_len, embed_dim = x.shape
        residual = x
        # 转换为 (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)
        
        # 多头自注意力
        attn_output, _ = self.self_attn(x, x, x)
        
        # 转换回 (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(0, 1)
        
        # 通过MLP输出
        return self.out_proj(attn_output + residual) + residual


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, ff_dim=1024, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 使用 PyTorch 的 MultiheadAttention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # MLP
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),  # 使用更节省显存的激活函数
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

    def forward(self, x):
        # batch_size, seq_len, embed_dim = x.shape
        residual = x
        # 转换为 (seq_len, batch_size, embed_dim)
        # x = x.transpose(0, 1)
        
        # 多头自注意力
        attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        
        # 转换回 (batch_size, seq_len, embed_dim)
        # attn_output = attn_output.transpose(0, 1)
        
        # 通过MLP输出
        return self.out_proj(self.norm2(attn_output)) + residual






class BaseSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 使用 PyTorch 的 MultiheadAttention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # 转换为 (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)
        
        # 多头自注意力
        attn_output, _ = self.self_attn(x, x, x)
        
        # 转换回 (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(0, 1)
        
        return attn_output

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),  # 使用更节省显存的激活函数
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.out_proj(x)

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = BaseSelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多头自注意力模块
        attn_output = self.self_attn(x)
        attn_output = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络模块
        ff_output = self.feed_forward(attn_output)
        ff_output = self.norm2(attn_output + self.dropout(ff_output))
        
        return ff_output

class LayerMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.transformer_layer1 = TransformerLayer(embed_dim, num_heads, ff_dim, dropout)
        self.transformer_layer2 = TransformerLayer(embed_dim, num_heads, ff_dim, dropout)

    def forward(self, x):
        # 第一层Transformer
        x = self.transformer_layer1(x)
        # 第二层Transformer
        x = self.transformer_layer2(x)
        return x




class SSMAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, ff_dim=1024, num_layers=2):
        super().__init__()

        # �| ~F�~O| �~Z个 Mamba2 模�~]~W
        self.self_attn_layers = nn.ModuleList([
            Mamba(
                d_model=embed_dim,  # 模�~^~K维度
                d_state=16,         # SSM �~J��~@~A�~I��~U�~[| �~P
                d_conv=4,           # �~@�~C��~M�积宽度
                expand=2,           # �~]~W�~I��~U�~[| �~P
            ) for _ in range(num_layers)
        ])

        # �~H~]�~K�~L~V MLP
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),  # 使�~T��~[��~J~B�~\~A�~X��~X�~Z~D�~@活�~G��~U�
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        # �~@~Z�~G�~O�~@�~B�~Z~D Mamba2 模�~]~W
        residual = x
        for layer in self.self_attn_layers:
            x = layer(x)  # �~K差�~^�~N�

        # �~@~Z�~G MLP �~S�~G�
        return self.out_proj(x + residual)


class INP_Former(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
            ss=None,
    ) -> None:
        super(INP_Former, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]
        self.ss = ss if ss else torch.nn.Identity()


        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0


    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    def forward(self, x_):
        x = self.encoder.prepare_tokens(x_)
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)

        vit_x = x.clone()


        agg_prototype = self.prototype_token
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)


        g_loss = self.gather_loss(x, agg_prototype)

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        # ae_output = self.ss(x_)
        # vit_x = vit_x.permute(0, 2, 1).contiguous()
        # b, embed, _ = vit_x.shape 
        # vit_x = vit_x.view(b, embed, side, side)
        # ae_output = self.ss(vit_x)
        # ae_output = self.ss(vit_x)

        # ae_outputes = []
        # for blk in self.ss:
        #     vit_x = blk(vit_x, vit_x)
        #     ae_outputes.append(vit_x)
        # for i, ae_output in  enumerate(ae_outputes):
            
        #     ae_output = ae_output.permute(0, 2, 1).contiguous()
        #     b, embed, _ = ae_output.shape 
        #     ae_output = ae_output.view(b, embed, side, side)
        #     ae_outputes[i] = ae_output


        ae_outputes = []
        ae_output =  vit_x.clone()
        for blk in self.ss:
            ae_output = blk(vit_x, ae_output) + vit_x
            ae_outputes.append(ae_output)
        for i, ae_output in  enumerate(ae_outputes):
            
            ae_output = ae_output.permute(0, 2, 1).contiguous()
            b, embed, _ = ae_output.shape 
            ae_output = ae_output.view(b, embed, side, side)
            ae_outputes[i] = ae_output


        return en, de, ae_outputes, g_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)









































