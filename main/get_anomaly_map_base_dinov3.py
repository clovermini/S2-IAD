"""Anomaly-map generation using CLIP with DINOv3 intermediate features.

This script builds text-guided anomaly maps, augments them with DINOv3-based
similarity cues, and evaluates the final results on the target dataset.
"""

import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(os.path.join(o_path, '../'))

import time
import json
import torch
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
import random
import argparse
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms
from models import open_clip
from models.dinov2.models.vision_transformer import vit_large
from few_shot import memory_surgery
from dataset import datasets
from utils import visualizer
from metrics import metrics
from tqdm import tqdm
from utils.tools import *
from logging import getLogger
from prompt_ensemble import prepare_text_feature
from similarity_calculation import *


def setup_seed(seed):  # Set random seeds.
    """Set random seeds for reproducible runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class CLIP_AD(nn.Module):
    """Wrapper for CLIP and the external DINOv3 backbone."""

    def __init__(self, model_name = 'ViT-B-16-plus-240', pretrain = 'laion400m_e32', img_size=240, device='cuda'):
        super(CLIP_AD, self).__init__()

        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name, pretrained=pretrain, force_image_size=img_size)

        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.device = device

        pretrained_model_name = "/data/datasets/pub/public/bigmodel_weights/dinov3"  # "facebook/dinov3-vitl16-pretrain-lvd1689m"
        self.dino_model = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map="auto", 
            trust_remote_code=True, local_files_only=True
        )
    
    @torch.no_grad()
    def encode_text(self, text, return_tokens=False):

        text = self.tokenizer(text, context_length=self.model.context_length).to(self.device)

        text_token, all_tokens = self.model.encode_text(text, return_tokens=return_tokens)
        text_token /= text_token.norm(dim=-1, keepdim=True)  
        if return_tokens:
            all_tokens /= all_tokens.norm(dim=-1, keepdim=True)  
            return text_token.float(), text, all_tokens.float()
        return text_token
    
    @torch.no_grad()
    def encode_image(self, image, feature_list=None, DPAM_layer=None, ignore_residual=False):   # Image encoding.
        """Encode images and return CLIP visual features."""
        b, _, _, _ = image.shape

        class_tokens, tokens, patch_tokens = self.model.encode_image(image, None, proj = True, feature_list = feature_list, DPAM_layer = DPAM_layer, ignore_residual = ignore_residual)  # feature_list = [3, 6, 9, 12], DPAM_layer = 10
        return class_tokens, tokens, patch_tokens
    
    @torch.no_grad()
    def get_dino_features(self, image):
        with torch.no_grad():
            outputs = self.dino_model(image, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        mid_tokens = []
        for mid_layer in [6, 12, 18, 24]:
            mid_tokens.append(hidden_states[mid_layer][:, 5:, :])
        return mid_tokens
    

@torch.no_grad()
def test(args,):
    """Run the DINOv3-enhanced anomaly detection pipeline."""
    img_size = args.image_size
    patch_size = args.patch_size   # 14  # 16
    feature_list = args.feature_list   # [3,6,9]
    dpam_layer = args.dpam_layer
    if dpam_layer < 1:
        dpam_layer = None
    print('feature_list ', feature_list, ' dpam_layer ', dpam_layer)
    dataset_dir = args.data_path
    des_path = args.des_path
    meta_path = args.meta_path
    save_path = args.save_path
    dataset_name = args.dataset
    k_shot = args.k_shot
    surgery_type = args.surgery_type
    if '_res' in surgery_type:
        ignore_residual = False   # args.ignore_residual
        surgery_type = surgery_type.replace('_res', '')
    else:  # ignore_residual 
        ignore_residual = True
    print('surgery_type ', surgery_type, ' ignore_residual ', ignore_residual)
    visualize = args.visualize
    save_anomaly_map = args.save_anomaly_map
    use_detailed = args.use_detailed
    batch_sim_topk = args.batch_sim_topk
    self_sim_topk = args.self_sim_topk

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    txt_path = os.path.join(save_path, 'log.txt')
    logger = set_logger(txt_path)

    print('**************** args ***************')
    for k,v in sorted(vars(args).items()):
        logger.info("%s", str(k)+' = '+str(v))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIP_AD(args.model, args.pretrained, img_size=img_size, device=device)   # model_name
    model.to(device)
    model.model.visual.DAPM_replace(DPAM_layer = dpam_layer, surgery_type=surgery_type)   # clip surgery

    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),   # img_size 240
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    
    preprocess = model.preprocess

    preprocess.transforms[0] = transforms.Resize(size=(img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                 max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(img_size, img_size))

    # Select the class list that should be evaluated under the current dataset mode.
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'metal_own':
        obj_list = ['casting_billet', 'steel_pipe', 'KolektorSDD', 'KolektorSDD2']        

    datasets.CLSNAMES = obj_list
    CLSNAMES_map_index = {}
    for k, index in zip(obj_list, range(len(obj_list))):
        CLSNAMES_map_index[k] = index
    datasets.CLSNAMES_map_index = CLSNAMES_map_index
    test_data = datasets.MetalDataset(root=dataset_dir, meta_path=meta_path, transform=preprocess, target_transform=transform, mode='test', k_shot=k_shot, save_dir=save_path, obj_name=obj_list)
    print('******* running ... obj_list ', obj_list)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=False)  # 32

    model.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []  # image level text_probs
    results['pr_sp'] = [] # image level label
    
    ########################################
    if k_shot == 0:
        few = False
    else:
        few = True

    with torch.no_grad(): 
        Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features, Mem_redundant_features = prepare_text_feature(model, obj_list, des_path, use_detailed)

        print('############ few_shot ', few, ' k_shot ', args.k_shot)
        if few:
            mem_features = memory_surgery(model.to(device), obj_list, des_path, preprocess, args.k_shot, device, feature_list, dpam_layer, ignore_residual)

        for index, items  in enumerate(tqdm(test_dataloader)):
            images = items['img'].to(device)
            cls_name = items['cls_name']
            cls_id = items['cls_id']
            results['cls_names'].extend(cls_name)
            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            results['imgs_masks'].append(gt_mask)  # px
            results['gt_sp'].extend(items['anomaly'].detach().cpu())

            b, c, h, w = images.shape   # [32, 3, 240, 240]
  
            average_normal_features = Mermory_avg_normal_text_features[cls_id]
            average_anomaly_features = Mermory_avg_abnormal_text_features[cls_id]
            text_features = torch.cat((average_normal_features, average_anomaly_features), dim = 1)
  
            image_features, tokens, patch_features = model.encode_image(images, feature_list, dpam_layer, ignore_residual)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
            text_probs = compute_score(image_features, text_features.permute(0, 2, 1))   # [24, 1, 2]
            text_probs = text_probs[:, 0, 1]  # z0score  # softmax already models the normal/abnormal distribution  text_probs[:, 0, 1]  [bs, 1, 2]

            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx != (len(patch_features)-1):  # Only use the patch features from the last layer.
                    continue
                patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)   # [32, 226, 640]

                similarity = compute_sim_minus(patch_feature, text_features.permute(0, 2, 1)) # [:,:,1]   # [32, 169]
                similarity_map = get_similarity_map(similarity, args.image_size)  # [24, 15, 15, 2] / [24, 37, 37, 2]
                anomaly_map = similarity_map[...,1]  # similarity_map[...,1]   [24, 15, 15, 2] / [24, 37, 37, 2]
                anomaly_map_list.append(anomaly_map)

            anomaly_map_text = torch.stack(anomaly_map_list)
            anomaly_map_text = anomaly_map_text.mean(dim = 0)

            # patch_features self sim to mean
            dino_features = model.get_dino_features(images)
            
            ## compute self-similarity maps
            anomaly_map_list_self = []
            for idx, patch_feature in enumerate(dino_features):  # patch_features
                if idx == 0:  # Use patch features from selected layers only.
                    continue
                self_sim_map = get_self_sim(patch_feature)
                anomaly_map_list_self.append(self_sim_map)
        
            anomaly_map_self = torch.stack(anomaly_map_list_self).mean(dim=0)

            # get batch sim
            anomaly_map_list_batch_sim = []
            for idx, patch_feature in enumerate(dino_features):  # patch_features  dino_features
                patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)   # [32, 226, 640]
                similarity_map = get_batch_sim(patch_feature, k=batch_sim_topk, reduction='mean')
                anomaly_map_list_batch_sim.append(similarity_map)
        
            anomaly_map_batch_sim = torch.stack(anomaly_map_list_batch_sim).mean(dim=0)
            anomaly_map = anomaly_map_batch_sim
            # anomaly_map = anomaly_map_self
            text_probs_anomaly_map = get_topk_mean(anomaly_map, 1)
        
            if few:  # Adapted from VAND-APRIL-GAN.
                anomaly_maps_few_shot = []
                for idx, p in enumerate(dino_features):  # Use all patch features  patch_features  dino_features
                    cos = few_shot(mem_features, p, cls_name, idx)
                    side = int(p.shape[1] ** 0.5)
                    anomaly_map_few_shot = cos.reshape((b, side, side)).cuda()
                    anomaly_maps_few_shot.append(anomaly_map_few_shot.cpu().numpy())
                anomaly_map_few_shot = np.mean(anomaly_maps_few_shot, axis=0)
                anomaly_map_few_shot = torch.from_numpy(anomaly_map_few_shot).to(device)
                anomaly_map = (anomaly_map + anomaly_map_few_shot) 

                text_probs_anomaly_map = (text_probs_anomaly_map + get_topk_mean(anomaly_map_few_shot, 1))
            text_probs = text_probs_anomaly_map

            anomaly_map_final = F.interpolate(torch.tensor(anomaly_map).unsqueeze(1), size=img_size, mode='bilinear', align_corners=True)
            anomaly_map_final = anomaly_map_final.squeeze(1)
            results['pr_sp'].extend(text_probs.detach().cpu())
            results['anomaly_maps'].append(anomaly_map_final)

            # Visualization.
            if visualize and k_shot == 0:
                show_path = os.path.join(save_path, 'vis/') 
                if not os.path.exists(show_path):
                    os.mkdir(show_path)
                visualizer.vis(items['img_path'], anomaly_map_final, text_probs.detach().cpu(), img_size, show_path, items['cls_name'], gt_mask.squeeze(1))

            if save_anomaly_map and k_shot == 4:
            
                anomaly_map = anomaly_map.cpu().numpy()
                for idx, path in enumerate(items['img_path']):
                    cls_name = items['cls_name'][idx]
                    image_name = path.split('/')[-1]
                    path = os.path.join(save_path, 'anomaly_map/', cls_name)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    anomaly_save_path = os.path.join(path, image_name.split('.')[0]+'_anomaly.npy')

                    ano_map = anomaly_map[idx]
                    np.save(anomaly_save_path, ano_map)
                    print('saving ... anomaly_save_path ', anomaly_save_path)

        results['imgs_masks'] = torch.cat(results['imgs_masks'])
        results['anomaly_maps'] = torch.cat(results['anomaly_maps']).detach().cpu().numpy()
        print('anomaly_maps ', results['anomaly_maps'].shape, ' imgs_masks ', results['imgs_masks'].shape)

        st_time = time.time()
        metric_results = metrics.cal_metrics(obj_list, results)
        logger.info("\n%s", metric_results)
        print('cal_metrics costs ', time.time()-st_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/test', help='path to save results')
    parser.add_argument("--des_path", type=str, default='', help='path to defect description')
    parser.add_argument("--meta_path", type=str, default='', help='path to data')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--feature_list", type=int, nargs="+", default=[3, 6, 9, 12], help="features used")   # [3, 6, 9, 12], DPAM_layer = 10
    parser.add_argument("--dpam_layer", type=int, default=10, help="surgery layer")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--patch_size", type=int, default=16, help="image size")
    parser.add_argument("--batch_sim_topk", type=int, default=24, help="topk for batch sim")
    parser.add_argument("--self_sim_topk", type=int, default=10, help="topk for self sim")
    parser.add_argument("--k_shot", type=int, default=10, help="10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")

    parser.add_argument("--surgery_type", type=str, default="vv", help="clip surgery/clearclip")
    parser.add_argument("--use_detailed", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--save_anomaly_map", action='store_true')

    args = parser.parse_args()

    setup_seed(args.seed)
    test(args)
