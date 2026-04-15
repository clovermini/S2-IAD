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
from models.dinov2.models.vision_transformer import vit_large  # 确保 dinov2 目录在 PYTHONPATH
from few_shot import memory_surgery
from dataset import datasets
from utils import visualizer
from metrics import metrics
from tqdm import tqdm
from logging import getLogger
from prompt_ensemble import prepare_text_feature

# from open_clip import get_tokenizer   # tokenizer


def setup_seed(seed):  # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# model_name = 'ViT-B-16-plus-240'  pretrain = 'laion400m_e32'  default  /home/data/liuchuni/.cache/clip/vit_b_16_plus_240-laion400m_e32-699c4b84.pt
# model_name = 'ViT-L-14-quickgelu'   pretrained = 'dfn2b'
# model_name = 'ViT-B-16-SigLIP-384'  pretrained = 'webli' 
# model_name = 'ViT-L-14-336'  pretrain = 'openai'
# model_name = 'ViT-SO400M-14-SigLIP-384'  pretrain = 'webli'
# model_name = 'ViT-H-14-378-quickgelu'  pretrain = 'dfn5b'   # /home/data/liuchuni/.cache/huggingface/hub/models--apple--DFN5B-CLIP-ViT-H-14-378
# model_name = 'ViT-L-16-SigLIP-384' pretrain='webli'
class CLIP_AD(nn.Module):
    def __init__(self, model_name = 'ViT-B-16-plus-240', pretrain = 'laion400m_e32', img_size=240, device='cuda'):
        super(CLIP_AD, self).__init__()
        # model_name = 'ViT-H-14-378-quickgelu' 
        # pretrain = 'dfn5b'  #
        # pretrain = '/home/data/liuchuni/.cache/huggingface/hub/models--apple--DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin'  #
        # model_name = 'ViT-B-16-plus-240'  
        # pretrain = 'laion400m_e32'
        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name, pretrained=pretrain, force_image_size=img_size)

        # self.tokenizer = open_clip.tokenizer
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.device = device
        # print('tokenizer ', self.tokenizer)

        pretrained_model_name = "/data/datasets/pub/public/bigmodel_weights/dinov3"  # "facebook/dinov3-vitl16-pretrain-lvd1689m"
        # processor = AutoImageProcessor.from_pretrained(pretrained_model_name, trust_remote_code=True, local_files_only=True)
        self.dino_model = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map="auto", 
            trust_remote_code=True, local_files_only=True
        )

        '''
        # 1. 构造模型
        self.dino_model = vit_large(patch_size=14,
                  img_size=518,
                  init_values=1.0,
                  block_chunks=0,
                  num_register_tokens=4)   # reg4 版本

        # 2. 加载本地权重
        ckpt_path = '/data/account/liuchuni/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth'
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.dino_model.load_state_dict(state_dict, strict=True)
        self.dino_model = self.dino_model.cuda().eval()
        '''
    
    @torch.no_grad()
    def encode_text(self, text, return_tokens=False):
        # from open_clip import tokenizer
        # text = self.tokenizer.tokenize(text)
        text = self.tokenizer(text, context_length=self.model.context_length).to(self.device)

        # print('encode_text text ', text.shape, ' self.model.context_length ', self.model.context_length)
        text_token, all_tokens = self.model.encode_text(text, return_tokens=return_tokens)
        #print('encode_text text_token ', text_token.shape)
        text_token /= text_token.norm(dim=-1, keepdim=True)  
        if return_tokens:
            all_tokens /= all_tokens.norm(dim=-1, keepdim=True)  
            return text_token.float(), text, all_tokens.float()
        return text_token
    
    @torch.no_grad()
    def encode_image(self, image, feature_list=None, DPAM_layer=None, ignore_residual=False):   # 图像编码
        #print('encode_image image shape ', image.shape)   # [32, 3, 240, 240]
        b, _, _, _ = image.shape

        class_tokens, tokens, patch_tokens = self.model.encode_image(image, None, proj = True, feature_list = feature_list, DPAM_layer = DPAM_layer, ignore_residual = ignore_residual)  # feature_list = [3, 6, 9, 12], DPAM_layer = 10
        #print('encode_image patch_tokens ', len(patch_tokens)) 
        #for i,ft in enumerate(patch_tokens):
        #    print('patch_tokens ', i, ' -- ', ft.shape)  # base_8 [226, 32, 896]  # cls + patch token  [1, 0, 2]

        #print('encode_image class_tokens ', class_tokens.shape)   # [32, 640] 
        #print('encode_image tokens ', tokens.shape)   # [32, 729, 1024]] 
        return class_tokens, tokens, patch_tokens
    
    @torch.no_grad()
    def get_dino_features(self, image):
        with torch.no_grad():
            outputs = self.dino_model(image, output_hidden_states=True)
        #mid_tokens = self.dino_model.get_intermediate_layers(image, n=[5, 11, 17, 23])
        # The intermediate features are now in `outputs.hidden_states`
        hidden_states = outputs.hidden_states
        # print(f"Number of hidden states (embedding layer + {len(hidden_states) - 1} transformer blocks): {len(hidden_states)}")
        mid_tokens = []
        for mid_layer in [6, 12, 18, 24]:
            mid_tokens.append(hidden_states[mid_layer][:, 5:, :])
        '''
        # You can now access the features from any layer
        # For example, to get the output of the last layer:
        last_layer_features = hidden_states[-1]
        print(f"Shape of last layer features: {last_layer_features.shape}")

        # To get the output of the first transformer block (index 1):
        first_block_features = hidden_states[1]
        print(f"Shape of first block's features: {first_block_features.shape}")
        '''
        return mid_tokens
    

def compute_score(image_features, text_features):  # 计算 图像文本相似得分
    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    text_probs = (torch.bmm(image_features.unsqueeze(1), text_features)/0.07).softmax(dim=-1)

    return text_probs


def compute_sim(image_features, text_features):  # 计算 图像文本相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    simmarity = (torch.bmm(image_features.squeeze(2), text_features)/0.07).softmax(dim=-1)
    return simmarity


def compute_sim_minus(image_features, text_features):  # 计算 图像文本相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    simmarity = torch.bmm(image_features.squeeze(2), text_features)

    normal_sim = simmarity[:, :, 0].clone()
    abnormal_sim = simmarity[:, :, 1].clone()
    abnormal_sim = abnormal_sim - normal_sim  # + 0.15
    # print('abnormal_sim max ', abnormal_sim.max(), ' min ', abnormal_sim.min())
    abnormal_sim[abnormal_sim < 0] = 0
    simmarity[:, :, 1] += abnormal_sim
    # simmarity_minus = torch.cat([normal_sim, abnormal_sim], dim=0)
    # simmarity_minus = simmarity_minus.unsqueeze(0).permute(0, 2, 1)
    simmarity_softmax = (simmarity/0.07).softmax(dim=-1)
    return simmarity_softmax


def get_similarity_map(sm, shape):
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    #sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm


def few_shot(memory, token, class_name, idx):
    retrive = []
    for i in class_name:
        L, N, D = memory[i][idx].shape   # [980, 1, 640]    [5, 225, 640]
        # print('few-shot ... class_name ', class_name, 'idx ', idx, ' -- ', memory[i][idx].shape)
        retrive.append(memory[i][idx].permute(2, 1, 0).reshape(D,-1)) # D NL   # [640, 225*5]
    retrive = torch.stack(retrive)# B D NL   # [32, 640, 225*5]   
    # print('retrive ', retrive.shape, ' token ', token.shape)
    # B D L    [32, 169, 640]  [32, 640, 225*5]   
    M = 1/2 * torch.min(1.0 - torch.bmm(F.normalize(token.squeeze(2), dim = -1), F.normalize(retrive, dim = 1)), dim = -1)[0]
    return M


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def get_self_sim(patch_feature):
    '''
    自相似异常图
    patch_feature: torch.Size([5, 1369, 768])
    '''
    B, N, D = patch_feature.shape
    H = W = int(N ** 0.5)
    # print('patch_feature.shape ', patch_feature.shape, ' -- ', H)

    # 1. L2 归一化
    patch_feature = F.normalize(patch_feature, dim=-1)  # [B, N, D]

    # 2. 计算两两余弦相似度矩阵 (1 - cosθ) -> [N, N]
    sim_matrix = torch.bmm(patch_feature, patch_feature.transpose(1, 2))  # [B, N, N]

    global_sim = sim_matrix.mean(dim=1)  # [B, N] 每个patch与全局的相似度

    global_sim = 1 - global_sim.view(B, H, W)
    
    return global_sim


def get_self_sim_topk_batch(patch_feature, initial_anomaly_map, k=5, reduction='mean'):
    """
    自相似异常图：每个 patch 与本图内最相似的 top-k 个 patch 的距离，并利用 top-k 分值更新初始异常得分图
    :param patch_feature: Tensor [B, N, D]，B=batch size，N=patch 数量，D=特征维度
    :param initial_anomaly_map: Tensor [B, N]，初始异常得分图
    :param k: int，取 top-k 最近邻
    :param reduction: 'mean' 或 'min'，如何聚合 top-k 分值
    :return:
        anomaly_map: Tensor [B, H, W]，基于 top-k 距离的异常得分图
        updated_anomaly_map: Tensor [B, H, W]，更新后的异常得分图
    """
    B, N, D = patch_feature.shape
    H = W = int(N ** 0.5)  # 假设 patch 排布为正方形

    # 1. L2 归一化
    patch_feature = F.normalize(patch_feature, dim=-1)  # [B, N, D]

    # 2. 计算两两余弦相似度矩阵 (1 - cosθ) -> [B, N, N]
    sim_matrix = torch.bmm(patch_feature, patch_feature.transpose(1, 2))  # 余弦相似度
    dist_matrix = 1.0 - sim_matrix  # 余弦距离

    # 3. 排除自身 patch（对角线）
    identity_matrix = torch.eye(N, device=patch_feature.device).unsqueeze(0).repeat(B, 1, 1)
    dist_matrix = dist_matrix + identity_matrix * 1e6

    # 4. 对每个 patch，找到 top-k 最近邻的距离和索引
    topk_dist, topk_indices = torch.topk(dist_matrix, k=k, largest=False, dim=-1)  # [B, N, k]

    # 5. 聚合 top-k 距离
    if reduction == 'mean':
        anomaly_score = topk_dist.mean(dim=-1)  # [B, N]
    elif reduction == 'min':
        anomaly_score = topk_dist.min(dim=-1)[0]  # [B, N]
    else:
        raise ValueError("reduction must be 'mean' or 'min'")
    # 6. reshape 回 (B, H, W) 得到异常得分图
    anomaly_map = anomaly_score.view(B, H, W)  # [B, H, W]

    if initial_anomaly_map is None:
        return anomaly_map, None

    # 7. 更新初始异常得分图
    # 将 top-k 分值从 initial_anomaly_map 中提取出来并聚合
    print('initial_anomaly_map ', initial_anomaly_map.size())
    initial_anomaly_map_flat = initial_anomaly_map.view(B, -1)  # [B, N]
    assert initial_anomaly_map_flat.shape[-1] == topk_indices.shape[1]
    # 检查索引范围
    # Gather top-k scores using advanced indexing
    batch_indices = torch.arange(B, device=patch_feature.device).view(B, 1, 1).expand(B, N, k)
    topk_initial_scores = initial_anomaly_map_flat[batch_indices, topk_indices]  # [B, N, k]
    print('initial_anomaly_map_flat ', initial_anomaly_map_flat.size(), ' topk_indices ', topk_indices.size(), ' anomaly_map ', anomaly_map.size())
    
    if reduction == 'mean':
        updated_initial_scores = topk_initial_scores.mean(dim=-1)  # [B, N]
    elif reduction == 'min':
        updated_initial_scores = topk_initial_scores.min(dim=-1)[0]  # [B, N]

    # 8. 更新后的异常得分图
    updated_anomaly_map = updated_initial_scores.view(B, H, W)  # [B, H, W]

    return anomaly_map, updated_anomaly_map


def get_batch_sim(batch_patch_features, k=5, reduction='mean'):
    """
    自相似异常图：每张图像 vs batch 内其余所有图像
    :param batch_patch_features: Tensor [B, N, D]
    :param k: int, top-k 最近邻
    :param reduction: 'mean' or 'min' 如何把 k 个距离汇成一个分数
    :return: Tensor [B, H, W],  H=W=int(sqrt(N))
    """
    B, N, D = batch_patch_features.shape
    H = W = int(N ** 0.5)

    # 1. L2 归一化
    x = F.normalize(batch_patch_features, dim=-1)          # [B, N, D]

    # 2. 构造“除自己外”的 mask
    mask = ~torch.eye(B, device=x.device, dtype=torch.bool)  # [B, B]
    # 后续 reshape 成 [B, 1, B*N] 广播用

    # 3. 把 batch 内所有 patch 拼在一起 [B*N, D]
    all_patches = x.view(-1, D)  # [B*N, D]

    # 4. 两两余弦距离矩阵 (1-cos)  -> [B*N, B*N]
    sim = torch.mm(all_patches, all_patches.T)  # 余弦相似度
    dist = 1.0 - sim                            # 余弦距离

    # 5. 去掉自身图像的 patch（对角块）
    mask_full = mask.repeat_interleave(N, dim=0).repeat_interleave(N, dim=1)
    dist = dist.masked_fill(~mask_full, 1e6)    # 把自身图像距离置为很大值

    # 6. 每个 query patch 取 top-k 最小距离
    topk_dist, _ = torch.topk(dist, k=k, largest=False, dim=-1)  # [B*N, k]

    # 7. 聚合 k 个距离
    if reduction == 'mean':
        score = topk_dist.mean(dim=-1)  # [B*N]
    elif reduction == 'min':
        score = topk_dist.min(dim=-1)[0]
    else:
        raise ValueError("reduction must be 'mean' or 'min'")

    # 8. reshape 回 (B, H, W)
    anomaly_map = score.view(B, H, W)
    # print('get_batch_sim anomaly_map ', anomaly_map.size())
    return anomaly_map


def get_topk_mean(anomaly_map, k=1):
    # topk avg
    # 1. 展平 anomaly_map
    anomaly_map_flat = anomaly_map.view(anomaly_map.shape[0], -1)  # [B, H*W]
    # 2. 提取每个样本的 top-k 值
    topk_values, _ = torch.topk(anomaly_map_flat, k, dim=-1)  # [B, k]
    topk_mean = topk_values.mean(dim=-1)  # [B]
    return topk_mean


def set_logger(txt_path):
    import logging
     # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='a+')  # w
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


@torch.no_grad()
def test(args,):
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

    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        test_data = datasets.MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                 aug_rate=-1, mode='test', obj_name=obj_list)
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        test_data = datasets.VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='test', obj_name=obj_list)
    elif dataset_name == 'metal_own':
        # obj_list = ['BSD_cls', 'DAGM2007_Class10', 'neu_rail', 'aluminum', 'steel_rail', 'moderately_thick_plates', 'DAGM2007_Class9', 'wood', 'Marbled', 'Mesh', 'cold_rolled_strip_steel', 'severstal_steel', 
        #            'hot_rolled_strip_annealing_picking', 'Perforated', 'DAGM2007_Class7', 'Stratified', 'AITEX', 'steel_pipe', 'Blotchy', 'BTech_02', 'bao_steel', 'Matted', 'KolektorSDD', 'BSData', 'medium_heavy_plate', 
        #            'aluminum_strip', 'DAGM2007_Class1', 'DAGM2007_Class6', 'leather', 'aluminum_ingot', 'neu_leather', 'DAGM2007_Class3', 'tianchi_aluminum', 'neu_aluminum', 'wide_thick_plate', 'gc10_steel_plate', 'Woven_127', 
        #            'rail_surface', 'neu_tile', 'Magnetic_tile', 'metal_plate', 'DAGM2007_Class4', 'Woven_068', 'grid', 'KolektorSDD2', 'Woven_104', 'road_crack', 'Woven_001', 'DAGM2007_Class8', 'neu_hot_rolled_strip', 
        #            'hot_rolled_strip_steel', 'neu_magnetic_tiles', 'Fibrous', 'neu_steel', 'DAGM2007_Class5', 'Woven_125', 'DAGM2007_Class2', 'ssgd_glasses', 'wukuang_medium_plate', 'nan_steel', 'tile']
        # obj_list = ['al_224_light']
        # obj_list = ['KolektorSDD2', 'steel_pipe', 'casting_billet']
        # obj_list = ['casting_billet', 'steel_pipe']
        # obj_list = ['KolektorSDD', 'KolektorSDD2']
        obj_list = datasets.CLSNAMES
        test_data = datasets.MetalDataset(root=dataset_dir, meta_path=meta_path, transform=preprocess, target_transform=transform, mode='test', k_shot=k_shot, save_dir=save_path, obj_name=obj_list)
        # obj_list = test_data.get_cls_names()
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
            #print('images shape ', images.shape)  
  
            average_normal_features = Mermory_avg_normal_text_features[cls_id]
            average_anomaly_features = Mermory_avg_abnormal_text_features[cls_id]
            # redundant_features = Mem_redundant_features[cls_id]
        
            # text_features = torch.cat((average_normal_features - redundant_features, average_anomaly_features - redundant_features), dim = 1)
            text_features = torch.cat((average_normal_features, average_anomaly_features), dim = 1)
            #print('text_features ', text_features.shape)
  
            image_features, tokens, patch_features = model.encode_image(images, feature_list, dpam_layer, ignore_residual)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #print('image_features ', image_features.shape, ' ', image_features[0])
        
            text_probs = compute_score(image_features, text_features.permute(0, 2, 1))   # [24, 1, 2]
            text_probs = text_probs[:, 0, 1]  # z0score  # softmax 就考虑了正常和异常的分布  text_probs[:, 0, 1]  [bs, 1, 2]
            # print('text_probs ', text_probs.shape, ' ', text_probs)
            #sys.exit()

            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx != (len(patch_features)-1):  # 只用最后一层的patch_features
                    continue
                patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)   # [32, 226, 640]

                # similarity = compute_sim(patch_feature, text_features.permute(0, 2, 1)) # [:,:,1]   # [32, 169]
                similarity = compute_sim_minus(patch_feature, text_features.permute(0, 2, 1)) # [:,:,1]   # [32, 169]

                #sim_min = similarity.min(dim=1, keepdim=True)[0]   # [32, 1]
                #sim_max = similarity.max(dim=1, keepdim=True)[0]   # [32, 1]
                # 防止除零
                #similarity = (similarity - sim_min) / (sim_max - sim_min + 1e-8)  # [32, 169]
                
                # print('similarity ', similarity.size())
                similarity_map = get_similarity_map(similarity, args.image_size)  # [24, 15, 15, 2] / [24, 37, 37, 2]
                # print('path text ... similarity ', similarity.shape, ' similarity_map ', similarity_map.shape) # [24, 1370, 2]  [24, 37, 37, 2]
                # print('similarity_map ', similarity_map.shape)  # [24, 15, 15, 2] / [24, 37, 37, 2]
                anomaly_map = similarity_map[...,1]  # similarity_map[...,1]   [24, 15, 15, 2] / [24, 37, 37, 2]
                #redundant_feats = similarity_map.mean(3, keepdim=False) # along cls dim
                #print('redundant_feats ', redundant_feats.shape)
                #anomaly_map = similarity_map[...,1] - redundant_feats
                anomaly_map_list.append(anomaly_map)

            anomaly_map_text = torch.stack(anomaly_map_list)
            #print('anomaly_map v0 ', anomaly_map.shape)
            anomaly_map_text = anomaly_map_text.mean(dim = 0)

            # text_probs_anomaly_map = get_topk_mean(anomaly_map_text, 1)

            # print('anomaly_map ', anomaly_map.max().item(), ' anomaly_map_self ', anomaly_map_self.max().item())

            # patch_features self sim to mean
            dino_features = model.get_dino_features(images)
            
            ## get self sim
            anomaly_map_list_self = []
            for idx, patch_feature in enumerate(dino_features):  # patch_features
                if idx == 0:  # 用指定层patch_features
                    continue
                # patch_feature = patch_feature[:,1:, :]
                
                #patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)   # [32, 226, 640]
                #mean_patch = patch_feature.mean(dim=1, keepdim=True)  # [32, 1, 640]
                ##print('mean_patch ', mean_patch.shape)
                #similarity = 1-torch.matmul(patch_feature, mean_patch.permute(0, 2, 1))
                ## print('similarity ', similarity.shape)
                #similarity_map = get_similarity_map(similarity, args.image_size)
                ##print('path mean ... similarity ', similarity.shape, ' similarity_map ', similarity_map.shape) # [24, 1370, 1] [24, 37, 37, 1]
                #anomaly_map_list_self.append(similarity_map[...,0])
                
                self_sim_map = get_self_sim(patch_feature)
                anomaly_map_list_self.append(self_sim_map)
        
            anomaly_map_self = torch.stack(anomaly_map_list_self).mean(dim=0)
            # anomaly_map_list.append(torch.stack(anomaly_map_list_self).mean(dim=0))
            

            # get batch sim
            anomaly_map_list_batch_sim = []
            anomaly_map_list_self_topk = []
            simmarity_softmax_text_update_list = []
            for idx, patch_feature in enumerate(dino_features):  # patch_features  dino_features
                #if idx <= 2:  # 用指定层patch_features
                #    continue
                # patch_feature = patch_feature[:,1:, :]
                patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)   # [32, 226, 640]
                similarity_map = get_batch_sim(patch_feature, k=batch_sim_topk, reduction='mean')
                # print('similarity_map ', similarity_map.size())
                anomaly_map_list_batch_sim.append(similarity_map)
                '''
                if idx == 0:
                    continue
                simmarity_softmax_text = None
                if idx == 3:
                    simmarity_softmax_text = anomaly_map_text 
                sim_topk, simmarity_softmax_text_update = get_self_sim_topk_batch(patch_feature, simmarity_softmax_text, k=self_sim_topk, reduction='mean')
                anomaly_map_list_self_topk.append(sim_topk)
                if simmarity_softmax_text_update is not None:
                    simmarity_softmax_text_update_list.append(simmarity_softmax_text_update)
                '''
        
            anomaly_map_batch_sim = torch.stack(anomaly_map_list_batch_sim).mean(dim=0)
            #anomaly_map_list.append(torch.stack(anomaly_map_list_self).mean(dim=0))

            #anomaly_map_self_topk = torch.stack(anomaly_map_list_self_topk).mean(dim=0)
            #anomaly_map_text_update = torch.stack(simmarity_softmax_text_update_list).mean(dim=0)
            
            # anomaly_map = (anomaly_map_text + anomaly_map_batch_sim + anomaly_map_self) / 3.0
            # anomaly_map = (anomaly_map_text + anomaly_map_batch_sim + anomaly_map_self_topk) / 3.0
            # anomaly_map = (anomaly_map_text + anomaly_map_self) / 2.0
            anomaly_map = anomaly_map_batch_sim
            #anomaly_map = (anomaly_map_text_update + anomaly_map_self) / 2.0
            '''
            print_min_max(anomaly_map_text, 'anomaly_map_text')
            print_min_max(anomaly_map_text_update, 'anomaly_map_text_update')
            print_min_max(anomaly_map_batch_sim, 'anomaly_map_batch_sim')
            print_min_max(anomaly_map_self, 'anomaly_map_self')
            print_min_max(anomaly_map_update, 'anomaly_map_update')
            print_min_max(anomaly_map, 'anomaly_map')
            print('text_probs min:', text_probs.min().item(), 'max:', text_probs.max().item())
            '''
            #anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = 4)) for i in anomaly_map.detach().cpu()], dim = 0 )
            #print('anomaly_map ', anomaly_map.shape)
            text_probs_anomaly_map = get_topk_mean(anomaly_map, 1)
        
            if few:  # 结合 VAND-APRIL-GAN 改写一下
                anomaly_maps_few_shot = []
                for idx, p in enumerate(dino_features):  # 用全部的patch features  patch_features  dino_features
                    # p = p[:, 1:, :]  # 去除 cls

                    cos = few_shot(mem_features, p, cls_name, idx)
                    #anomaly_map_few_shot = np.min((1 - cos), 0).reshape(1, 1, height, height)
                    side = int(p.shape[1] ** 0.5)
                    anomaly_map_few_shot = cos.reshape((b, side, side)).cuda()
                    anomaly_maps_few_shot.append(anomaly_map_few_shot.cpu().numpy())
                anomaly_map_few_shot = np.mean(anomaly_maps_few_shot, axis=0)
                anomaly_map_few_shot = torch.from_numpy(anomaly_map_few_shot).to(device)
                anomaly_map = (anomaly_map + anomaly_map_few_shot) 
                # anomaly_map = anomaly_map.to(device)

                text_probs_anomaly_map = (text_probs_anomaly_map + get_topk_mean(anomaly_map_few_shot, 1))

                #print_min_max(anomaly_map_few_shot, 'anomaly_map_few_shot')
                #print_min_max(anomaly_map, 'anomaly_map')
                # max anomaly 没有对anomaly 归一化，因此这个可能会占主导位置 ，理论上应该按照类别整体归一化一下
            
                # text_probs = (text_probs.cpu() + torch.max(torch.max(anomaly_map, dim = 1)[0],dim = 1)[0])/2.0   # [32]
                #print('text_probs final min:', text_probs.min().item(), 'max:', text_probs.max().item())

            # 原始方案
            #text_probs = (text_probs + torch.max(torch.max(anomaly_map, dim = 1)[0],dim = 1)[0])/2.0   # [32]
            
            # text_probs = (text_probs + text_probs_anomaly_map) / 2.0
            text_probs = text_probs_anomaly_map
            
            '''
            # anomaly-aware representation
            last_patch_feature = tokens # patch_features[-1][:, 1:, :]
            # print('last_patch_feature ', last_patch_feature.size())
            anomaly_map_flat = anomaly_map.view(anomaly_map.shape[0], -1)  # [B, N]
            anomaly_map_softmax = F.softmax(anomaly_map_flat, dim=-1)  # [B, N]
            weighted_patch_feature = last_patch_feature * anomaly_map_softmax.unsqueeze(-1)  # [B, N, D]
            global_anomaly_representation = weighted_patch_feature.mean(dim=1)  # [B, D]
            global_anomaly_representation = F.normalize(global_anomaly_representation, dim=-1)  # [B, D]
            text_probs_patch = compute_score((global_anomaly_representation + image_features) / 2.0, text_features.permute(0, 2, 1))   # [24, 1, 2]
            text_probs_patch = text_probs_patch[:, 0, 1]
            #print('text_probs_patch ', text_probs_patch.shape, ' ', text_probs_patch)
            '''

            anomaly_map_final = F.interpolate(torch.tensor(anomaly_map).unsqueeze(1), size=img_size, mode='bilinear', align_corners=True)
            anomaly_map_final = anomaly_map_final.squeeze(1)
            # print('anomaly_map_final ', anomaly_map_final.size())
            results['pr_sp'].extend(text_probs.detach().cpu())
            results['anomaly_maps'].append(anomaly_map_final)

            # 可视化
            if visualize and k_shot == 0:
                show_path = os.path.join(save_path, 'vis/')  # '/home/data/liuchuni/projects/fsad_big_model/defect_lvlms/output/anomaly_maps_surgery_base_show/'  # _test
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
