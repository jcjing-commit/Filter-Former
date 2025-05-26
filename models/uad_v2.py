import torch
import torch.nn as nn
import torch.nn.functional as F
import math



from torch import nn
from torchvision.datasets import ImageFolder

# def get_autoencoder(out_channels=384, side=64):
#     return nn.Sequential(
#         # encoder
#         nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
#                   padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
#                   padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
#                   padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
#                   padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
#                   padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
#         # decoder
#         nn.Upsample(size=3, mode='bilinear'),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
#                   padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=8, mode='bilinear'),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
#                   padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=15, mode='bilinear'),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
#                   padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=32, mode='bilinear'),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
#                   padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=63, mode='bilinear'),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
#                   padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=127, mode='bilinear'),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
#                   padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=side, mode='bilinear'),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
#                   padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
#                   stride=1, padding=1)
#     )


def get_autoencoder_v1(out_channels=384, side=64):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2,
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
        nn.Upsample(size=128, mode='bilinear'),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=128, mode='bilinear'),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=side, mode='bilinear'),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=3,
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

def get_autoencoder(out_channels=384, in_channel=768, side=64):
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
        vit_x = vit_x.permute(0, 2, 1).contiguous()
        b, embed, _ = vit_x.shape 
        vit_x = vit_x.view(b, embed, side, side)
        ae_output = self.ss(vit_x)

        return en, de, ae_output, g_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)









































