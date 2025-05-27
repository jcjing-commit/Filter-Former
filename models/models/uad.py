import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2

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
        self.ss = ss

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0


    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    def forward(self, x_):
        x = self.encoder.prepare_tokens(x_)   # x shape bs,wh+cls+reg(4),embeding
        
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

        # tmp_x = self.ss(x_)
        filter_states = self.ss(x)  # x shape b,hw, embdding; 
        # tmp_x = self._process_foreground(tmp_x)

        # s_loss = self.gather_loss(x, hidden_states) 

        distance_st = (x - filter_states)**2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])


        agg_prototype = self.prototype_token
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)
        g_loss = self.gather_loss(x, agg_prototype)   #x为encoder的输出，agg_prototype为聚合后的prototype

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

        en = [e - filter_states for e in en]
        de = [d - filter_states for d in de]


        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        filter_states = filter_states.permute(0,2,1).reshape([x.shape[0], -1, side, side]) 
        return en, de, filter_states, g_loss, loss_hard

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def _process_foreground(self, fg):
        batch_size, _, height, width = fg.shape
        # Define the filter size and kernel
        filter_size = 2
        kernel = np.ones((filter_size, filter_size), np.uint8)
        # Create an empty array to store the morphological operation results
        final_masks = np.zeros_like(fg)

        # Iterate over the batch of images
        for i in range(batch_size):
            mask = fg[i, 0]  # Get the individual image
            mask = (mask*255).astype(np.uint8)
            _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
            # Apply the morphological operation to the image
            mask_morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = max(contours, key=cv2.contourArea)
            final_mask = np.zeros_like(mask_morph)
            cv2.drawContours(final_mask, [max_contour], -1, 1, thickness=cv2.FILLED)

            # Store the result in the corresponding index of the output array
            final_masks[i, 0] = final_mask

        return final_masks
    







































