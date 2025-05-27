import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger, evaluation_batch_vis_ZS, evaluation_batch_vis_ZS_v3

# Dataset-Related Modules
from dataset import MVTecDataset, RealIADDataset, MVTecDataset2, MVTecDataset2_test
from dataset import get_data_transforms, check_mvtec2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

import torch.nn.functional as F

# Model-Related Modules
from models import vit_encoder
from models.uad_v3 import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
# from models.selective_scan import Selective_Scan, buildMambaIRv2Base, buildMambaIRv2_light#, buildMambaIRv2_tiny
# from models.selective_scan import buildMambaIRv2_tiny
from models.uad import get_autoencoder,MultiHeadSelfAttention
import json

warnings.filterwarnings("ignore")
def main(args):
    # Fixing the Random Seed
    setup_seed(1)
    # Data Preparation
    data_transform, gt_transform, infer_transforms = get_data_transforms(args.input_size, args.crop_size)

    if args.thresholded_cfg is not None:
        with open(args.thresholded_cfg, 'r') as f:
            thresholded_cfg = json.load(f)
    else:
        thresholded_cfg = None


    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA':
        train_path = os.path.join(args.data_path, args.item, 'train')
        test_path = os.path.join(args.data_path, args.item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    elif args.dataset == 'Real-IAD' :
        train_data = RealIADDataset(root=args.data_path, category=args.item, transform=data_transform, gt_transform=gt_transform,
                                    phase='train')
        test_data = RealIADDataset(root=args.data_path, category=args.item, transform=data_transform, gt_transform=gt_transform,
                                   phase="test")
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    elif args.dataset == 'MVTec-AD2' :

        data_root = check_mvtec2(args.data_path)
        assert data_root is not None, "Failed to download MVTec-AD2 dataset."
        args.data_path = data_root


        train_path = os.path.join(args.data_path, args.item, 'train')
        test_path = os.path.join(args.data_path, args.item)

        if args.phase == 'train':
            phase = 'test_public'
        else:
            phase = args.phase

        train_data = ImageFolder(root=train_path, transform=data_transform)
        test_data = MVTecDataset2_test(root=test_path, transform=infer_transforms, gt_transform=gt_transform, phase=phase)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16,
                                                       drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16)


    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder info
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(args.INP_num, embed_dim))
                     for _ in range(1)])

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)
    ss = MultiHeadSelfAttention(embed_dim, num_heads)
  
    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP, ss=ss)

    model = model.to(device)

    if args.phase == 'train':
        # Model Initialization
        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=args.total_epochs*len(train_dataloader),
                                           warmup_iters=50)
        print_fn('train image number:{}'.format(len(train_data)))

        os.makedirs(os.path.join(args.save_dir, args.save_name, args.item), exist_ok=True)
        # Train
        max_pf1 = 0
        best_map_name = None
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                en, de, ae_output, g_loss  = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)

                # 将列表中的张量堆叠成一个新的张量
                en_stacked = torch.stack(en)
                de_stacked = torch.stack(de)

                # 计算差
                diff = de_stacked - en_stacked
                mean_diff = torch.mean(diff, dim=0)
                ae_loss = global_cosine_hm_adaptive([ae_output], [mean_diff], y=3)
                loss = loss + 0.2 * g_loss + ae_loss

                # mse_loss = F.mse_loss(ae_output, mean_diff) 
                # loss = loss + 0.2 * g_loss + 0.33*mse_loss       


                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'model.pth'))
            print_fn('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list)))

            if (epoch + 1) % 1 == 0:
                results, map_name = evaluation_batch_vis_ZS_v3(model, test_dataloader, device, max_ratio=0.01, resize_mask=256, save_root=args.work_out, phase="test_public", thresholded_cfg=thresholded_cfg)
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                if f1_px > max_pf1:
                    max_pf1 = f1_px
                    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'model_best.pth'.format(epoch+1,f1_px)))
                    best_map_name = map_name


        results, _  = evaluation_batch_vis_ZS_v3(model, test_dataloader, device, max_ratio=0.01, resize_mask=args.crop_size, save_root=args.work_out, phase="test_public", thresholded_cfg=thresholded_cfg, target_map=best_map_name)

        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
        
        return results, best_map_name
    else:  # Test
         
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, args.item, 'model_best.pth')), strict=True)
        model.eval()
     
        # anomaly_map_tag = {
        #                     "can":"anomaly_map","fabric":"anomaly_map_aee_de",
        #                     "fruit_jelly":"anomaly_map_eadda_d","rice":"anomaly_map_eadda_d",
        #                     "sheet_metal":"anomaly_map","vial":"merge_map",
        #                     "wallplugs":"anomaly_map","walnuts":"anomaly_map",
        #                     }
        # anomaly_map_tag = None


        save_path = os.path.join(args.save_dir, 'best_map_record.json')

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            with open(save_path, 'r+') as f:
                anomaly_map_tag = json.load(f)

        assert anomaly_map_tag is not None, "anomaly_map_tag is None"   
        assert args.item in anomaly_map_tag, f"{args.item} not in anomaly_map_tag"  

        target_map = None
        if anomaly_map_tag:
            target_map = anomaly_map_tag[args.item]


        results, _ = evaluation_batch_vis_ZS_v3(model, test_dataloader, device, max_ratio=0.01, resize_mask=896, save_root=args.work_out, phase=args.phase, thresholded_cfg=thresholded_cfg, target_map=target_map)
        return results, None


if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD2') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'../mvtec_ad_2')  # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='INP-Former-Single-Class')

    #
    parser.add_argument('--work_out', type=str, default='./workout')
    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_large_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=896)
    parser.add_argument('--crop_size', type=int, default=896)
    parser.add_argument('--INP_num', type=int, default=16)

    # training info
    parser.add_argument('--total_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--phase', type=str, default='train')

    parser.add_argument('--thresholded_cfg', type=str, default=None, help='thresholded_cfg path for testing')
  

    args = parser.parse_args()
    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'   #TODO

    # category info
    if args.dataset == 'MVTec-AD':
        # args.data_path = 'E:\IMSN-LW\dataset\mvtec_anomaly_detection' # '/path/to/dataset/MVTec-AD/'
        args.item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    elif args.dataset == 'VisA':
        # args.data_path = r'E:\IMSN-LW\dataset\VisA_pytorch\1cls'  # '/path/to/dataset/VisA/'
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset == 'Real-IAD':
        # args.data_path = 'E:\IMSN-LW\dataset\Real-IAD'  # '/path/to/dataset/Real-IAD/'
        args.item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    elif args.dataset == 'MVTec-AD2':
        args.item_list = ["rice", "walnuts", "fabric", "fruit_jelly", "sheet_metal", "vial", "wallplugs",
                            "can"]
        args.item_list = ["rice", "walnuts",]


    result_list = []
    best_map_record = {}
    for item in args.item_list:
        args.item = item
        print("=============================",args.phase,": ", item)

        result, best_map_name = main(args)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = result
        result_list.append([args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px])
        if best_map_name is not None:
            best_map_record[item] = best_map_name

        save_path = os.path.join(args.save_dir, 'best_map_record.json')

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            with open(save_path, 'r+') as f:
                data = json.load(f)
                data.update(best_map_record)
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
        else:
            with open(save_path, 'w') as f:
                json.dump(best_map_record, f, indent=4)

    mean_auroc_sp = np.mean([result[1] for result in result_list])
    mean_ap_sp = np.mean([result[2] for result in result_list])
    mean_f1_sp = np.mean([result[3] for result in result_list])

    mean_auroc_px = np.mean([result[4] for result in result_list])
    mean_ap_px = np.mean([result[5] for result in result_list])
    mean_f1_px = np.mean([result[6] for result in result_list])
    mean_aupro_px = np.mean([result[7] for result in result_list])

    print_fn(result_list)
    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            mean_auroc_sp, mean_ap_sp, mean_f1_sp,
            mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px))
