import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2


class LearnableFeatureFilter(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, mode='adaptive'):
        """
        可学习特征过滤网络
        Args:
            in_channels: 输入特征通道数
            reduction_ratio: 通道压缩比例
            mode: 'adaptive'(自适应阈值), 'gating'(门控), 'sparse'(稀疏化)
        """
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 自适应阈值预测
        if mode == 'adaptive':
            self.threshold_predictor = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels // reduction_ratio, 1, 1),
                nn.Sigmoid()
            )
        
        # 稀疏化控制参数
        self.sparsity_controller = nn.Parameter(torch.tensor(0.5)) if mode == 'sparse' else None
        
    def forward(self, x):
        # 输入形状: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 通道注意力权重
        channel_weights = self.channel_attention(x)  # [B, C, 1, 1]
        
        # 空间注意力权重
        spatial_weights = self.spatial_attention(x)  # [B, 1, H, W]
        
        if self.mode == 'adaptive':
            # 预测每个空间位置的动态阈值
            relative_threshold = self.threshold_predictor(x)  # [B, 1, H, W]
            abs_x = torch.abs(x)
            global_threshold = torch.quantile(abs_x.view(B, -1), q=0.85, dim=1).view(B, 1, 1, 1)
            threshold = global_threshold * relative_threshold
            
            # 软过滤
            mask = torch.sigmoid((abs_x - threshold) * 10)  # 10是温度系数，可学习
            output = x * mask * channel_weights * spatial_weights
            
        elif self.mode == 'gating':
            # 门控机制
            gate = torch.sigmoid(x * channel_weights * spatial_weights)
            output = x * gate
            
        elif self.mode == 'sparse':
            # 可学习稀疏化
            abs_x = torch.abs(x)
            k = torch.sigmoid(self.sparsity_controller)  # 学习保留特征的比例
            kth = int((1 - k) * H * W)
            
            # 每个通道独立处理
            output = []
            for i in range(B):
                for j in range(C):
                    vals = abs_x[i,j].flatten()
                    if kth > 0:
                        kth_val = torch.kthvalue(vals, kth).values
                        mask = (abs_x[i,j] >= kth_val).float()
                    else:
                        mask = torch.ones_like(abs_x[i,j])
                    output.append(x[i,j] * mask * channel_weights[i,j] * spatial_weights[i])
            
            output = torch.stack(output).view(B, C, H, W)
        
        return output




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

        # 定义可学习的 alpha 参数
        # 定义可学习的 alpha 参数
        self.alpha = nn.Parameter(torch.tensor([0.1] * len(fuse_layer_decoder)*2).reshape((2,-1)))

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        
        # 修改统计量存储方式，使用detach()防止梯度计算
        self.register_buffer('normal_mean', None)
        self.register_buffer('normal_std', None)
        self.filter_strength = nn.Parameter(torch.tensor(1.0))  # 可学习的过滤强度
        
        # 添加显存优化标志
        self.memory_efficient = True

    def update_normal_stats(self, features):
        # 确保不计算梯度
        with torch.no_grad():
            # 计算当前batch的统计量，使用detach()分离计算图
            current_mean = features.mean(dim=[0,1], keepdim=True).detach()
            current_std = features.std(dim=[0,1], keepdim=True).detach()
            
            # 更新全局统计量
            if self.normal_mean is None or self.normal_std is None:
                self.normal_mean = current_mean
                self.normal_std = current_std
            else:
                # 使用移动平均更新统计信息
                if self.memory_efficient:
                    # 更节省显存的方式：原地操作
                    self.normal_mean.mul_(0.9).add_(0.1 * current_mean)
                    self.normal_std.mul_(0.9).add_(0.1 * current_std)
                else:
                    self.normal_mean = 0.9 * self.normal_mean + 0.1 * current_mean
                    self.normal_std = 0.9 * self.normal_std + 0.1 * current_std

    def filter_features(self, features):
        # 确保使用detach()防止梯度传播到统计量
        with torch.no_grad():
            normal_mean = self.normal_mean.detach()
            normal_std = self.normal_std.detach()
        
        # 计算特征与正常统计的偏差
        z_scores = (features - normal_mean) / (normal_std + 1e-6)
        anomaly_scores = torch.sigmoid(self.filter_strength * z_scores.abs())
        
        # 创建过滤权重 (1表示保留，0表示过滤)
        filter_weights = 1.0 - anomaly_scores
        return features * filter_weights


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

        

        agg_prototype = self.prototype_token
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)
        g_loss = self.gather_loss(x, agg_prototype)   #x为encoder的输出，agg_prototype为聚合后的prototype

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        # xx = x.clone()
        # #d_hard = torch.quantile(xx, q=0.85)   #TODO 
        # #xx[xx <= d_hard] = 1e-6
        # abs_vals = torch.abs(xx)
        # mean_val = abs_vals.mean()
        # std_val = abs_vals.std()

        # # 使用高斯函数计算权重
        # weights = torch.exp(-0.5 * ((abs_vals - mean_val) / (std_val + 1e-6)) ** 2)
        # xx = xx * weights


        if self.training:
            self.update_normal_stats(x)
        else:
            x = self.filter_features(x)


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

        # en = [e - self.alpha[0][i]*filter_states for i,e in enumerate(en)]
        # de = [d - self.alpha[1][i]*filter_states for i,d in enumerate(de)]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        # filter_states = filter_states.permute(0,2,1).reshape([x.shape[0], -1, side, side]) 
        # return en, de, filter_states, g_loss
        return en, de, g_loss

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
    







































