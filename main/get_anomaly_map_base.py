"""Zero-shot and few-shot anomaly map generation with CLIP and DINO features.

This script prepares the dataset, builds text prototypes, extracts image
features, and evaluates anomaly maps for each category.
"""

import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(os.path.join(o_path, '../'))

import time
import torch
import torch.nn as nn
import argparse
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms
from models import open_clip
from models.dinov2.models.vision_transformer import vit_large 
from few_shot import memory_surgery
from prompt_ensemble import prepare_text_feature
from similarity_calculation import *
from dataset import datasets
from utils import visualizer
from utils.tools import set_logger, setup_seed
from metrics import metrics
from tqdm import tqdm



class CLIP_AD(nn.Module):
    """Wrapper around the CLIP image/text encoder and the auxiliary DINO backbone."""

    def __init__(self, model_name = 'ViT-B-16-plus-240', pretrain = 'laion400m_e32', img_size=240, device='cuda'):
        super(CLIP_AD, self).__init__()

        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name, pretrained=pretrain, force_image_size=img_size)

        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.device = device

        # 1. 构造模型
        self.dino_model = vit_large(patch_size=14,
                  img_size=518,
                  init_values=1.0,
                  block_chunks=0,
                  num_register_tokens=4)   # reg4 版本

        # 2. 加载本地权重
        ckpt_path = '/XXX/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth'
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.dino_model.load_state_dict(state_dict, strict=True)
        self.dino_model = self.dino_model.cuda().eval()
    
    @torch.no_grad()
    def encode_text(self, text, return_tokens=False):
        """Encode text prompts and return normalized text embeddings."""

        text = self.tokenizer(text, context_length=self.model.context_length).to(self.device)

        text_token, all_tokens = self.model.encode_text(text, return_tokens=return_tokens)
        text_token /= text_token.norm(dim=-1, keepdim=True)  
        if return_tokens:
            all_tokens /= all_tokens.norm(dim=-1, keepdim=True)  
            return text_token.float(), text, all_tokens.float()
        return text_token
    
    @torch.no_grad()
    def encode_image(self, image, feature_list=None, DPAM_layer=None, ignore_residual=False): 
        """Encode images and return CLS tokens, token embeddings, and patch features."""
        # print('encode_image image shape ', image.shape)   # [32, 3, 240, 240]
        b, _, _, _ = image.shape

        class_tokens, tokens, patch_tokens = self.model.encode_image(image, None, proj = True, feature_list = feature_list, DPAM_layer = DPAM_layer, ignore_residual = ignore_residual)  # feature_list = [3, 6, 9, 12], DPAM_layer = 10
        # print('encode_image patch_tokens ', len(patch_tokens)) 
        # for i,ft in enumerate(patch_tokens):
        #     print('patch_tokens ', i, ' -- ', ft.shape)  # base_8 [226, 32, 896]  # cls + patch token  [1, 0, 2]

        # print('encode_image class_tokens ', class_tokens.shape)   # [32, 640] 
        # print('encode_image tokens ', tokens.shape)   # [32, 729, 1024]] 
        return class_tokens, tokens, patch_tokens
    
    @torch.no_grad()
    def get_dino_features(self, image):
        """Extract intermediate DINO features used for few-shot memory."""
        mid_tokens = self.dino_model.get_intermediate_layers(image, n=[5, 11, 17, 23])
        return mid_tokens


@torch.no_grad()
def test(args,):
    """Run the full evaluation loop and save anomaly-detection metrics."""
    img_size = args.image_size
    patch_size = args.patch_size   # 14 
    feature_list = args.feature_list   
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
    update_topk = args.update_topk
    score_topk = args.score_topk

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
    # k=0 means pure zero-shot inference; otherwise few-shot memory is enabled.
    if k_shot == 0:
        few = False
    else:
        few = True

    with torch.no_grad(): 
        # Prepare class-specific text prototypes before scanning the test set.
        Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features, _ = prepare_text_feature(model, obj_list, des_path, use_detailed)

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
            #print('images shape ', images.shape)  
  
            average_normal_features = Mermory_avg_normal_text_features[cls_id]
            average_anomaly_features = Mermory_avg_abnormal_text_features[cls_id]
        
            text_features = torch.cat((average_normal_features, average_anomaly_features), dim = 1)
            #print('text_features ', text_features.shape)
  
            image_features, _, patch_features = model.encode_image(images, feature_list, dpam_layer, ignore_residual)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #print('image_features ', image_features.shape, ' ', image_features[0])
        
            text_probs = compute_score(image_features, text_features.permute(0, 2, 1))   # [24, 1, 2]
            text_probs = text_probs[:, 0, 1]  # z0score  # softmax 就考虑了正常和异常的分布  text_probs[:, 0, 1]  [bs, 1, 2]
            # print('text_probs ', text_probs.shape, ' ', text_probs)

            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                # Only the last visual block is used to build the final anomaly map.
                if idx != (len(patch_features)-1): 
                    continue
                patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)   # [32, 226, 640]

                # similarity = compute_sim(patch_feature, text_features.permute(0, 2, 1)) # [:,:,1]   # [32, 169]
                similarity = compute_sim_minus(patch_feature, text_features.permute(0, 2, 1)) # [:,:,1]   # [32, 169]
                
                # print('similarity ', similarity.size())
                similarity_map = get_similarity_map(similarity, args.image_size)  # [24, 15, 15, 2] / [24, 37, 37, 2]
                # print('path text ... similarity ', similarity.shape, ' similarity_map ', similarity_map.shape) # [24, 1370, 2]  [24, 37, 37, 2]
                # print('similarity_map ', similarity_map.shape)  # [24, 15, 15, 2] / [24, 37, 37, 2]
                anomaly_map = similarity_map[...,1]  # similarity_map[...,1]   [24, 15, 15, 2] / [24, 37, 37, 2]

                anomaly_map_list.append(anomaly_map)

            anomaly_map_text = torch.stack(anomaly_map_list)
            anomaly_map_text = anomaly_map_text.mean(dim = 0)

            # patch_features self sim to mean
            dino_features = model.get_dino_features(images)
            
            # get self & batch sim
            anomaly_map_list_batch_sim = []
            anomaly_map_list_self_topk = []
            simmarity_softmax_text_update_list = []
            for idx, patch_feature in enumerate(dino_features):  # patch_features  dino_features

                patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)   # [32, 226, 640]
                similarity_map = get_batch_sim(patch_feature, k=batch_sim_topk, reduction='mean')
                anomaly_map_list_batch_sim.append(similarity_map)

                if idx == 0:
                    continue
                simmarity_softmax_text = None
                if idx == 3:
                    simmarity_softmax_text = anomaly_map_text 
                sim_topk, simmarity_softmax_text_update = get_self_sim_topk_batch(patch_feature, simmarity_softmax_text, k_self_sim=self_sim_topk, k_update=update_topk, reduction='mean')
                anomaly_map_list_self_topk.append(sim_topk)
                if simmarity_softmax_text_update is not None:
                    simmarity_softmax_text_update_list.append(simmarity_softmax_text_update)
        
            anomaly_map_batch_sim = torch.stack(anomaly_map_list_batch_sim).mean(dim=0)
            anomaly_map_self_topk = torch.stack(anomaly_map_list_self_topk).mean(dim=0)
            anomaly_map_text_update = torch.stack(simmarity_softmax_text_update_list).mean(dim=0)
            
            # for steel pipe
            fused_map = adaptive_fusion_spatial([anomaly_map_self_topk, anomaly_map_text_update], T=1.0)
            anomaly_map = adaptive_fusion_spatial([anomaly_map_batch_sim, fused_map], T=1.0)

            # for casting billet
            #fused_map = adaptive_fusion_spatial([anomaly_map_self_topk, anomaly_map_batch_sim], T=1.0)
            #anomaly_map = adaptive_fusion_spatial([anomaly_map_text_update, fused_map], T=1.0)

            # for ksdd
            #anomaly_map = adaptive_fusion_spatial([anomaly_map_text_update, anomaly_map_self_topk, anomaly_map_batch_sim], T=1.0)
            # or
            #fused_map = adaptive_fusion_spatial([anomaly_map_self_topk, anomaly_map_text_update], T=1.0)
            #anomaly_map = adaptive_fusion_spatial([anomaly_map_batch_sim, fused_map], T=1.0)

            # for mvtec
            #fused_map = adaptive_fusion_spatial([anomaly_map_self_topk, anomaly_map_text_update], T=1.0)
            #anomaly_map = (fused_map + anomaly_map_batch_sim) / 2.0
            
            # for visa
            #fused_map = adaptive_fusion_spatial([anomaly_map_self_topk, anomaly_map_text_update], T=1.0)
            #anomaly_map = (fused_map + anomaly_map_batch_sim) / 2.0

            text_probs_anomaly_map = get_topk_mean(anomaly_map, score_topk)
        
            if few:  # few-shot memory surgery is enabled, which requires additional DINO features and similarity calculations.
                anomaly_maps_few_shot = []
                for idx, p in enumerate(dino_features):  # 用全部的patch features  patch_features  dino_features

                    cos = few_shot(mem_features, p, cls_name, idx)
                    anomaly_map_few_shot = cos.reshape((b, h//patch_size, w//patch_size)).cuda()
                    anomaly_maps_few_shot.append(anomaly_map_few_shot.cpu().numpy())

                anomaly_map_few_shot = np.mean(anomaly_maps_few_shot, axis=0)
                anomaly_map_few_shot = torch.from_numpy(anomaly_map_few_shot).to(device)
                
                #anomaly_map = (anomaly_map + anomaly_map_few_shot)
                anomaly_map = adaptive_fusion_spatial([anomaly_map, anomaly_map_few_shot], T=1.0)

                text_probs_anomaly_map = (text_probs_anomaly_map + get_topk_mean(anomaly_map_few_shot, score_topk))


            #text_probs = (text_probs + torch.max(torch.max(anomaly_map, dim = 1)[0],dim = 1)[0])/2.0   # [32]            
            text_probs = (text_probs + text_probs_anomaly_map) / 2.0

            anomaly_map_final = F.interpolate(torch.tensor(anomaly_map).unsqueeze(1), size=img_size, mode='bilinear', align_corners=True)
            anomaly_map_final = anomaly_map_final.squeeze(1)
            # print('anomaly_map_final ', anomaly_map_final.size())
            results['pr_sp'].extend(text_probs.detach().cpu())
            results['anomaly_maps'].append(anomaly_map_final)

            # 可视化
            if visualize:   # and k_shot == 0:
                show_path = os.path.join(save_path, 'vis/')
                if not os.path.exists(show_path):
                    os.mkdir(show_path)
                visualizer.vis(items['img_path'], anomaly_map_final, text_probs.detach().cpu(), img_size, show_path, items['cls_name'], gt_mask.squeeze(1))

            if save_anomaly_map and k_shot == 4:
            
                anomaly_map = anomaly_map.cpu().numpy()
                #print('tokens ', tokens.shape)
                for idx, path in enumerate(items['img_path']):
                    cls_name = items['cls_name'][idx]
                    image_name = path.split('/')[-1]
                    path = os.path.join(save_path, 'anomaly_map/', cls_name)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    # token_save_path = path[:-4]+'_token.npy'
                    anomaly_save_path = os.path.join(path, image_name.split('.')[0]+'_anomaly.npy')

                    # token = tokens[idx]
                    ano_map = anomaly_map[idx]
                    # np.save(token_save_path, token)
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
    parser.add_argument("--update_topk", type=int, default=10, help="topk for update")
    parser.add_argument("--score_topk", type=int, default=1, help="topk for calc score")

    # parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")
    # few shot
    parser.add_argument("--k_shot", type=int, default=10, help="10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")

    parser.add_argument("--surgery_type", type=str, default="vv", help="clip surgery/clearclip")
    parser.add_argument("--use_detailed", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--save_anomaly_map", action='store_true')

    args = parser.parse_args()

    setup_seed(args.seed)
    test(args)
