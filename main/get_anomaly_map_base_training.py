import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(os.path.join(o_path, '../'))

import time
import torch
import json
import torch.nn as nn
import argparse
import numpy as np
from PIL import Image

import torch.nn.functional as F
import torchvision.transforms as transforms
from models import open_clip
from models.dinov2.models.vision_transformer import vit_large  # 确保 dinov2 目录在 PYTHONPATH
from few_shot import memory_surgery
from prompt_ensemble import prepare_text_feature
from similarity_calculation import *
from dataset import datasets
from utils import visualizer
from utils.tools import *
from metrics import metrics
from tqdm import tqdm


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, dynamic_margin=None):

        # Use the dynamic margin if it's provided, otherwise use the fixed one.
        margin_to_use = (dynamic_margin+self.margin) if dynamic_margin is not None else self.margin

        pos_distance = torch.sum((anchor - positive).pow(2), dim=-1)

        neg_distance = torch.sum((anchor - negative).pow(2), dim=-1)

        #print('pos_distance ', pos_distance.size())
        #print('neg_distance ', neg_distance.size())
        #print('pos_distance - neg_distance ', pos_distance - neg_distance)

        loss = torch.relu(pos_distance - neg_distance + margin_to_use)

        return torch.mean(loss)
    

# clip vv-attention model
class CLIP_AD(nn.Module):
    def __init__(self, model_name = 'ViT-B-16-plus-240', pretrain = 'laion400m_e32', img_size=240, device='cuda'):
        super(CLIP_AD, self).__init__()

        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name, pretrained=pretrain, force_image_size=img_size)

        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.device = device
    
    @torch.no_grad()
    def encode_text(self, text):

        text = self.tokenizer(text, context_length=self.model.context_length).to(self.device)

        text_token, _ = self.model.encode_text(text)
        text_token /= text_token.norm(dim=-1, keepdim=True)

        return text_token
    
    @torch.no_grad()
    def encode_image(self, image, feature_list=None, DPAM_layer=None, ignore_residual=False):   # 图像编码

        b, _, _, _ = image.shape
        class_tokens, tokens, patch_tokens = self.model.encode_image(image, None, proj = True, feature_list = feature_list, DPAM_layer = DPAM_layer, ignore_residual = ignore_residual)  # feature_list = [3, 6, 9, 12], DPAM_layer = 10

        return class_tokens, tokens, patch_tokens


class DualSoftPromptLearner(nn.Module):
    """
    This module learns two separate soft prompts:
    1. A 'normal_token' to represent the concept of a normal sample.
    2. An 'abnormal_token' to represent the concept of an abnormal sample.
    """
    def __init__(self, model, des_path, obj_name, num_prompt_tokens=2, use_detailed=True):
        super().__init__()
        self.clip_model = model
        self.tokenizer = model.tokenizer
        # self.num_prompt_tokens = num_prompt_tokens

        with open(des_path) as f:
            des=json.load(f)
        self.total_des = des
        self.dtype = self.clip_model.model.transformer.get_cast_dtype()
        embedding_dim = self.clip_model.model.token_embedding.weight.shape[1]  # 768
        #print('embedding_dim ', embedding_dim)

        des_info = self.total_des[obj_name]
        class_name = des_info['map']
        # "pipe_fryum": "pipe fryum",
        # "chewinggum": "chewing gum",
        if obj_name == 'pipe_fryum':
            class_name = 'pipe fryum'
        if obj_name == 'chewinggum':
            class_name = 'chewing gum'
        if 'pcb' in obj_name:
            class_name = 'printed circuit board'
        if obj_name == 'candle':
            class_name = 'candles'
        if obj_name == 'metal_nut':
            class_name = 'metal nut object'
        if obj_name == 'pill':
            class_name = 'white pill placed on a black background'  # 
        if obj_name == 'transistor':
            class_name = 'black transistor with white metal line'
        if obj_name == 'casting billet':
            class_name = 'steel industrial' # for casting billet
        print('map class_name ', class_name)

        # 1. Tokenize the actual class_name to get its token IDs
        with torch.no_grad():
            text_tokens = self.tokenizer(class_name).to(self.clip_model.device)
            # print('text_tokens ', text_tokens)
            
            # Isolate the tokens for the phrase itself, excluding Start-of-Text and End-of-Text tokens
            eot_token_index = torch.where(text_tokens[0] == self.tokenizer.eot_token_id)[0]

            phrase_token_ids = text_tokens[0, 1:eot_token_index]
            
            # The number of learnable tokens is now the length of our class name
            self.num_prompt_tokens = len(phrase_token_ids)
            print(f"Initializing with {self.num_prompt_tokens} learnable tokens.")

            # 2. Get the initial embeddings for these specific tokens from the embedding layer
            initial_embeddings = self.clip_model.model.token_embedding(phrase_token_ids).clone()
        
        # 3. Use this sequence of embeddings to initialize the learnable cls_token
        # The shape will be [1, num_prompt_tokens, embedding_dim] to support batching later
        self.cls_token = nn.Parameter(initial_embeddings.clone())
        print('self.cls_token ', self.cls_token.size())

        
        ## 1. Initialize TWO trainable soft prompt vectors
        #self.cls_token = nn.Parameter(
        #    torch.randn(num_prompt_tokens, embedding_dim, dtype=self.dtype)
        #)

        #cls_token = torch.empty(num_prompt_tokens, embedding_dim, dtype=self.dtype)
        #nn.init.normal_(cls_token, std=0.02)
        #self.cls_token = nn.Parameter(cls_token)  # to be optimized

        '''
        # 1. Get the embedding for your chosen initial word (e.g., "object")
        with torch.no_grad():
            # Tokenize the word and get its ID. We take the second token [0, 1] 
            # to skip the initial Start-of-Text token.
            init_token_id = self.tokenizer("object")[0, 1]
            
            # Get the corresponding embedding vector from the model's vocabulary
            initial_embedding = model.model.token_embedding.weight[init_token_id].clone()

        # 2. Use the fetched embedding to initialize the learnable tokens
        # We ensure the new parameters have the correct shape and are on the right device.
        if num_prompt_tokens > 1:
            # If using multiple tokens, repeat the initial embedding
            initial_embedding = initial_embedding.unsqueeze(0).repeat(num_prompt_tokens, 1)
        print('initial_embedding ', initial_embedding.size())
        self.cls_token = nn.Parameter(initial_embedding.unsqueeze(0).clone())
        '''

        # 冻结CLIP的所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # --- NEW: Define a unique placeholder word ---
        # self.placeholder_str = "zxy" 
        
        # Find the token ID for our placeholder word once.
        # self.placeholder_token_id = self.tokenizer(self.placeholder_str)[0, 1]
            
        placeholder_word = "zxy" 
        self.placeholder_str = " ".join([placeholder_word] * self.num_prompt_tokens)
        
        tokenized_placeholder = self.tokenizer(self.placeholder_str).to(self.clip_model.device)
        self.placeholder_token_ids = tokenized_placeholder[0, 1:1+self.num_prompt_tokens]

        # April GAN
        #self.state_normal_list = ['flawless {}']
        #self.state_anomaly_list = ['{} with flaw']
        self.state_normal_list = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
        self.state_anomaly_list = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        self.template_list = ['a cropped photo of the {}.']  # cropped photo of the

        self.inference_state_normal_list = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
        self.inference_state_anomaly_list = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        self.inference_state_normal_list_detailed = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
        self.inference_state_anomaly_list_detailed = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        #self.inference_template_list = self.template_list
        self.inference_template_list = ['a bad photo of a {}.', 
                              'a low resolution photo of the {}.', 
                              'a bad photo of the {}.', 
                              'a cropped photo of the {}.', 
                              'a bright photo of a {}.', 
                              'a dark photo of the {}.', 
                              'a photo of my {}.', 
                              'a photo of the cool {}.', 
                              'a close-up photo of a {}.', 
                              'a black and white photo of the {}.', 
                              'a bright photo of the {}.', 
                              'a cropped photo of a {}.', 
                              'a jpeg corrupted photo of a {}.', 
                              'a blurry photo of the {}.', 
                              'a photo of the {}.', 
                              'a good photo of the {}.', 
                              'a photo of one {}.', 
                              'a close-up photo of the {}.', 
                              'a photo of a {}.', 
                              'a low resolution photo of a {}.', 
                              'a photo of a large {}.', 
                              'a blurry photo of a {}.', 
                              'a jpeg corrupted photo of the {}.', 
                              'a good photo of a {}.', 
                              'a photo of the small {}.', 
                              'a photo of the large {}.', 
                              'a black and white photo of a {}.', 
                              'a dark photo of a {}.', 
                              'a photo of a cool {}.', 
                              'a photo of a small {}.', 
                              'there is a {} in the scene.', 
                              'there is the {} in the scene.', 
                              'this is a {} in the scene.', 
                              'this is the {} in the scene.', 
                              'this is one {} in the scene.']
        
        
        #if use_detailed:
        self.inference_state_normal_list_detailed.extend(des_info['des']['good'])
        self.inference_state_anomaly_list_detailed.extend(des_info['des']['defect'])

        self.state_normal_list  = self.inference_state_normal_list_detailed
        self.state_anomaly_list  = self.inference_state_anomaly_list_detailed
        
        #print('template_list ', len(self.template_list), ' inference_template_list ', len(self.inference_template_list))
        #print('state_normal_list ', len(self.state_normal_list), ' self.inference_state_normal_list ', len(self.inference_state_normal_list), ' inference_state_normal_list_detaied ', len(self.inference_state_normal_list_detaied))
        #print('state_anomaly_list ', len(self.state_anomaly_list), ' self.inference_state_anomaly_list ', len(self.inference_state_anomaly_list), ' inference_state_anomaly_list_detaied ', len(self.inference_state_anomaly_list_detaied))
        #print('inference_state_anomaly_list_detaied ', self.inference_state_anomaly_list_detaied)
        #print('inference_state_normal_list_detaied ', self.inference_state_normal_list_detaied)

        class_state = [ele.format(class_name) for ele in self.state_normal_list]
        normal_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
    
        class_state = [ele.format(class_name) for ele in self.state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
        print('prompt normal_ensemble_template ', len(normal_ensemble_template), ' anomaly_ensemble_template ', len(anomaly_ensemble_template))

        with torch.no_grad():
            normal_text_features = model.encode_text(normal_ensemble_template).float()
            abnormal_text_features = model.encode_text(anomaly_ensemble_template).float()

            self.normal_text_features = torch.mean(normal_text_features, dim = 0, keepdim= True) 
            self.normal_text_features /= self.normal_text_features.norm()
            self.abnormal_text_features = torch.mean(abnormal_text_features, dim = 0, keepdim= True)   # 取平均
            self.abnormal_text_features /= self.abnormal_text_features.norm()
        
        self.initial_check_done = False

    def _embed_and_encode(self, state_list, template_list):
        """Helper function to inject a token into templates and get features."""
        text_embeddings = self.clip_model.model.token_embedding

        prompt_features_list = []

        class_state = [ele.format(self.placeholder_str) for ele in state_list]
        ensemble_templates = [class_template.format(ele) for ele in class_state for class_template in template_list]
        
        '''
        if is_normal:   # handle the normal temples
            class_state = [ele.format(self.placeholder_str) for ele in self.state_normal_list]
            ensemble_templates = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
        else:
            class_state = [ele.format(self.placeholder_str) for ele in self.state_anomaly_list]
            ensemble_templates = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
        '''

        for template in ensemble_templates:
            
            text_tokens = self.tokenizer(template).to(self.clip_model.device)
            
            # --- FIX: Find the placeholder's index dynamically ---
            placeholder_index = torch.where(text_tokens[0] == self.placeholder_token_ids[0])[0]
            if placeholder_index.size(0) == 0:
                raise ValueError(f"Placeholder '{self.placeholder_str}' not found in template: '{template}'")
            placeholder_index = placeholder_index[0]
            #print('placeholder_index ', placeholder_index)
            #print('template ', template)
            #print('text_tokens ', text_tokens)

            end_placeholder_index = placeholder_index + self.num_prompt_tokens
            
            prefix_embeds = text_embeddings(text_tokens[:, :placeholder_index]).type(self.dtype)
            suffix_embeds = text_embeddings(text_tokens[:, end_placeholder_index:]).type(self.dtype)
            
            full_embeds = torch.cat([
                prefix_embeds,
                self.cls_token.unsqueeze(0),
                suffix_embeds
            ], dim=1)

            # print('full_embeds ', full_embeds.size(), ' text_tokens ', text_tokens.size())
            
            # --- REPLICATION OF encode_text ---
            x = full_embeds + self.clip_model.model.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            
            # 1. ADD THE ATTENTION MASK
            x = self.clip_model.model.transformer(x, attn_mask=self.clip_model.model.attn_mask)
            
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.model.ln_final(x)
            
            # 2. USE THE CORRECT POOLING METHOD
            # We need the original token IDs to determine where to pool from
            # For this manual pass, we need to create equivalent token IDs for the modified sequence
            pooled_x, _ = open_clip.text_global_pool(x, text_tokens, self.clip_model.model.text_pool_type)
            
            # 3. APPLY THE FINAL PROJECTION
            pooled_features = pooled_x @ self.clip_model.model.text_projection
            prompt_features_list.append(pooled_features)

        print('prompt_features_list ', len(prompt_features_list), ' -- ', prompt_features_list[0].size())
        # Average the features from all templates
        all_features = torch.cat(prompt_features_list)
        all_features = all_features / all_features.norm(dim=-1, keepdim=True)
        avg_features = all_features.mean(dim=0, keepdim=True)
        avg_features = avg_features / avg_features.norm(dim=-1, keepdim=True)

        '''
        # We only run this check once, on the first forward pass for the normal features.
        if not self.initial_check_done and is_normal:
            print("\n--- Running Initial State Verification ---")
            
            # Compare the dynamically generated features with the pre-calculated baseline
            are_close = torch.allclose(avg_features, self.normal_text_features, atol=1e-5)
            
            if are_close:
                print("✅ VERIFICATION PASSED: Initial dynamic features match the baseline.")
            else:
                print("❌ VERIFICATION FAILED: Initial features do not match.")
                # Optional: print values to see the difference
                print(f"   Similarity Score: {F.cosine_similarity(avg_features, self.normal_text_features).item()}")
                print(f"   L2 Distance: {torch.dist(avg_features, self.normal_text_features).item()}")

            print("--- Verification Complete ---\n")
            self.initial_check_done = True # Set the flag so we don't run this again
        # --- VERIFICATION LOGIC END ---
        '''
        
        return avg_features
    
    @torch.no_grad()
    def generate_final_features(self, use_detailed=False):
        """
        NEW METHOD: Uses the trained cls_token with the large, complex prompt lists
        to generate the final, high-quality text features.
        """
        print("\n--- Generating final features with complex prompt ensemble ---")
        # This method is almost identical to _embed_and_encode, but uses the inference lists
        
        if use_detailed:
            # Generate final normal features
            final_normal_features = self._embed_and_encode(self.inference_state_normal_list_detailed, self.inference_template_list)
        
            # Generate final anomaly features
            final_anomaly_features = self._embed_and_encode(self.inference_state_anomaly_list_detailed, self.inference_template_list)
        else:
            # Generate final normal features
            final_normal_features = self._embed_and_encode(self.inference_state_normal_list, self.inference_template_list)
        
            # Generate final anomaly features
            final_anomaly_features = self._embed_and_encode(self.inference_state_anomaly_list, self.inference_template_list)

        return final_normal_features, final_anomaly_features
    
    def forward(self, image_tensor, feature_list, dpam_layer, ignore_residual):  # image_tensor, feature_list, dpam_layer, ignore_residual
        # This part remains the same logic
 
        avg_positive_features = self._embed_and_encode(self.state_normal_list, self.template_list)
        avg_negative_features = self._embed_and_encode(self.state_anomaly_list, self.template_list)
        
        # 编码图像
        with torch.no_grad():
            cls_embedding, _, patch_tokens = self.clip_model.encode_image(image_tensor, feature_list, dpam_layer, ignore_residual)
            cls_embedding /= cls_embedding.norm(dim=-1, keepdim=True)
            patch_tokens = patch_tokens[-1]
            patch_tokens /= patch_tokens.norm(dim=-1, keepdim=True)
        
        return avg_positive_features, avg_negative_features, cls_embedding, patch_tokens

import torch.optim as optim

# --- 修改: train_adapter 函数现在会动态应用数据增强 ---
def train_adapter(model, preprocess, normal_images, img_size, save_path, logger, epochs, feature_list, dpam_layer, ignore_residual, device, lr=0.002, margin=0, reg_lambda=1):
    #optimizer = optim.AdamW([
    #    {'params': [model.compensation_embedding]},
    #    {'params': [model.max_t_norm_logit, model.max_t_anom_logit], 'lr': lr * 0.01}
    #], lr=lr)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=margin)
    
    # 1. 定义数据增强的流程
    # CLIP 自身的预处理包含 Resize 和 CenterCrop，可能会与 RandomResizedCrop 冲突。
    # 我们将 CLIP 的 Normalize 操作提取出来，与其他增强方法结合。
    
    # 从 model.preprocess 中提取 Normalize 参数
    clip_normalize = None
    for transform in preprocess.transforms:
        if isinstance(transform, transforms.Normalize):
            clip_normalize = transform
            break

    # 定义我们的训练增强流程
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        clip_normalize # 使用 CLIP 的归一化参数
    ])
    
    logger.info("\n--- Starting Training with Data Augmentation ---")
    logger.info(f"Augmentation pipeline: {train_transforms}")

    emb_ouput_path = os.path.join(save_path, 'adjusted_text_features/')  # '/home/data/liuchuni/projects/fsad_big_model/defect_lvlms/output/anomaly_maps_surgery_base_show/'  # _test
    if not os.path.exists(emb_ouput_path):
        os.mkdir(emb_ouput_path)
    logger.info(f"Embeddings will be saved to '{emb_ouput_path}/'")

    base_norm_path = os.path.join(emb_ouput_path, "base_normal_embedding.pt")
    base_anom_path = os.path.join(emb_ouput_path, "base_anomaly_embedding.pt")
    torch.save(model.normal_text_features.detach().cpu(), base_norm_path)
    torch.save(model.abnormal_text_features.detach().cpu(), base_anom_path)
    print(f"Saved base embeddings to {emb_ouput_path}/")

    
    for epoch in range(epochs):
        # 2. 在每个 epoch 的开始，动态生成一个增强后的图像批次
        # normal_images 是一个 PIL Image 对象的列表
        try:
            augmented_batch = torch.stack([train_transforms(img) for img in normal_images]).to(device)
        except TypeError:
            # 如果 CLIP 版本较老，ToTensor 可能已包含在 preprocess 中，需要调整
            # 这是一个更通用的兼容性写法
            base_transforms = transforms.Compose([t for t in preprocess.transforms if not isinstance(t, (transforms.Resize, transforms.CenterCrop, transforms.Normalize))])
            full_train_transforms = transforms.Compose([
                base_transforms, # 其他可能的变换如 ToTensor
                transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                clip_normalize
            ])
            augmented_batch = torch.stack([full_train_transforms(img) for img in normal_images]).to(device)

        optimizer.zero_grad()
        
        # 将增强后的图像输入模型

        avg_positive_features, avg_negative_features, cls_embedding, patch_tokens = model(augmented_batch, feature_list, dpam_layer, ignore_residual)

        print('cls_embedding ', cls_embedding.size(), ' patch_tokens ', patch_tokens.size(), ' avg_positive_features ', avg_positive_features.size(), ' avg_negative_features ', avg_negative_features.size())

        
        # compute mean
        mean_pos_learned = torch.mean(F.normalize(avg_positive_features, dim=-1), dim=0)
        mean_pos_handle = torch.mean(F.normalize(model.normal_text_features, dim=-1), dim=0)
        loss_match_pos = (mean_pos_handle - mean_pos_learned).norm(dim=0) ** 2.0

        mean_neg_learned = torch.mean(F.normalize(avg_negative_features, dim=-1), dim=0)
        mean_neg_handle = torch.mean(F.normalize(model.abnormal_text_features, dim=-1), dim=0)
        loss_match_neg = (mean_neg_handle - mean_neg_learned).norm(dim=0) ** 2.0

        print('loss_match_pos ', loss_match_pos, ' loss_match_neg ', loss_match_neg)

        cls_pos = torch.einsum('nc,cm->nm', cls_embedding, avg_positive_features.transpose(0, 1))
        cls_neg = torch.einsum('nc,cm->nm', cls_embedding, avg_negative_features.transpose(0, 1))
        print('cls_pos ', cls_pos.size(), ' cls_neg ', cls_neg.size())

        patch_pos = torch.einsum('nic,cj->nij', patch_tokens, avg_positive_features.transpose(0, 1))
        patch_neg = torch.einsum('nic,cj->nij', patch_tokens, avg_negative_features.transpose(0, 1))
        print('patch_pos ', patch_pos.size(), ' patch_neg ', patch_neg.size())

        logit_scale = model.clip_model.model.logit_scale

        logits_v2t_cls = torch.cat([cls_pos, cls_neg], dim=-1) * logit_scale
        target_v2t_cls = torch.zeros([logits_v2t_cls.shape[0]], dtype=torch.long).to(device)
        loss_v2t_cls = criterion(logits_v2t_cls, target_v2t_cls)
        print('logits_v2t_cls ', logits_v2t_cls.size(), ' target_v2t_cls ', target_v2t_cls.size(), ' loss_v2t_cls ', loss_v2t_cls)

        logits_v2t_patch = torch.cat([patch_pos, patch_neg], dim=-1) * logit_scale
        target_v2t_patch = torch.zeros([logits_v2t_patch.shape[0], logits_v2t_patch.shape[1]], dtype=torch.long).to(device)
        loss_v2t_patch = criterion(logits_v2t_patch.transpose(1, 2), target_v2t_patch)
        print('logits_v2t_patch ', logits_v2t_patch.size(), ' target_v2t_patch ', target_v2t_patch.size(), ' loss_v2t_patch ', loss_v2t_patch)

        
        # --- NEW: Calculate the per-image dynamic margin ---
        with torch.no_grad(): # This calculation should not be part of the gradient graph
            # For the CLS token
            dist_base_normal_cls = torch.sum((cls_embedding - model.normal_text_features).pow(2), dim=-1)
            dist_base_abnormal_cls = torch.sum((cls_embedding - model.abnormal_text_features).pow(2), dim=-1)
            dynamic_margin_cls = dist_base_abnormal_cls - dist_base_normal_cls
            dynamic_margin_cls = torch.clamp(dynamic_margin_cls, min=0)
            #print('dynamic_margin_cls ', dynamic_margin_cls)

            # For the patch tokens
            dist_base_normal_patch = torch.sum((patch_tokens - model.normal_text_features).pow(2), dim=-1)
            dist_base_abnormal_patch = torch.sum((patch_tokens - model.abnormal_text_features).pow(2), dim=-1)
            dynamic_margin_patch = dist_base_abnormal_patch - dist_base_normal_patch
            dynamic_margin_patch = torch.clamp(dynamic_margin_patch, min=0)
        
        print('dynamic_margin_cls ', dynamic_margin_cls.size(), ' dynamic_margin_patch ', dynamic_margin_patch.size())
        #print('dynamic_margin_patch ', dynamic_margin_patch)
        
        #dynamic_margin_cls = dynamic_margin_patch = None

        trip_loss_cls = criterion_tip(cls_embedding, avg_positive_features, avg_negative_features, dynamic_margin=dynamic_margin_cls)  # anchor, positive, negative
        trip_loss_patch = criterion_tip(patch_tokens, avg_positive_features, avg_negative_features, dynamic_margin=dynamic_margin_patch)  # anchor, positive, negative
        #print('trip_loss_cls ', trip_loss_cls, ' trip_loss_patch ', trip_loss_patch)
        
        # Calculate similarity with the specialized vector
        sim_pos_learned_cls = F.cosine_similarity(cls_embedding, avg_positive_features)
        
        # Calculate similarity with the generic vector (our new negative)
        sim_pos_base_cls = F.cosine_similarity(cls_embedding, model.normal_text_features)

        # Calculate similarity with the generic vector (our new negative)
        #sim_neg_cls = F.cosine_similarity(cls_embedding, avg_negative_features)
        
        # The loss function pushes sim_positive to be greater than sim_negative by a margin
        loss_margin_cls = torch.clamp(0 - sim_pos_learned_cls + sim_pos_base_cls, min=0).mean()  # 0.01 default
        #print('sim_learned ', sim_pos_learned_cls.size(), ' sim_base ', sim_pos_base_cls.size(), ' loss_margin ', loss_margin_cls)

        #trip_loss_cls = torch.clamp(margin - sim_pos_learned_cls + sim_neg_cls, min=0).mean()  # 0.01 default

        #loss_margin_cls = criterion_tip(cls_embedding, avg_positive_features, model.normal_text_features)

        # 1. Calculate similarity for each patch against the learned text feature
        # PyTorch will broadcast avg_positive_features to match the shape of patch_tokens
        sim_pos_learned_patch = F.cosine_similarity(patch_tokens, avg_positive_features, dim=-1)
        #sim_pos_neg_patch = F.cosine_similarity(patch_tokens, avg_negative_features, dim=-1)
        #trip_loss_patch = torch.clamp(margin - sim_pos_learned_patch + sim_pos_neg_patch, min=0).mean()  # 0.01 default

        # 2. Calculate similarity for each patch against the base text feature
        sim_base_patch = F.cosine_similarity(patch_tokens, model.normal_text_features, dim=-1)  # [4, 1369]

        # 3. Calculate the margin loss for the patches. 
        # The .mean() will average the loss across all patches in the batch.
        loss_margin_patch = torch.clamp(0 - sim_pos_learned_patch + sim_base_patch, min=0).mean()

        # loss_text_text = F.cosine_similarity(avg_positive_features, avg_negative_features).mean()

        # print('sim_learned_patch ', sim_learned_patch.size(), ' sim_base_patch ', sim_base_patch.size(), ' loss_margin_patch ', loss_margin_patch)
        
        # loss = loss_margin + (loss_v2t_cls ) + (trip_loss_cls ) # + (loss_match_pos+loss_match_neg) * reg_lambda
        #print('--------reg_lambda ', reg_lambda)
        #reg_lambda = 0
        loss = (loss_margin_cls + loss_margin_patch) + (loss_v2t_cls + loss_v2t_patch) + (trip_loss_cls + trip_loss_patch) + (loss_match_pos+loss_match_neg) * reg_lambda # reg_lambda # + loss_text_text
        #loss = (loss_margin_cls + loss_v2t_cls + trip_loss_cls) + (loss_match_pos+loss_match_neg) * reg_lambda # reg_lambda # + loss_text_text

        #loss = loss_margin_cls + loss_v2t_cls + trip_loss_cls  # + (loss_match_pos+loss_match_neg) * reg_lambda
        print('loss ', loss)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- 新增: 在每个epoch结束时保存张量 ---
        # 使用 .detach() 来分离计算图，只保存数据
        # 使用 .cpu() 是一个好习惯，可以使保存的文件不依赖于特定的GPU设备
        if ((epoch + 1) == epochs) or ((epoch + 1) % 10 == 0):
            norm_filename = os.path.join(emb_ouput_path, f"adjusted_normal_epoch_{epoch+1:04d}.pt")
            anom_filename = os.path.join(emb_ouput_path, f"adjusted_anomaly_epoch_{epoch+1:04d}.pt")

            # After the loop, call the new method to get the rich features
            final_normal_features, final_anomaly_features = model.generate_final_features(use_detailed=False)
    
            torch.save(final_normal_features.cpu(), norm_filename)
            torch.save(final_anomaly_features.cpu(), anom_filename)

            norm_filename = os.path.join(emb_ouput_path, f"adjusted_normal_epoch_{epoch+1:04d}_kead.pt")
            anom_filename = os.path.join(emb_ouput_path, f"adjusted_anomaly_epoch_{epoch+1:04d}_kead.pt")

            # After the loop, call the new method to get the rich features
            final_normal_features, final_anomaly_features = model.generate_final_features(use_detailed=True)
    
            torch.save(final_normal_features.cpu(), norm_filename)
            torch.save(final_anomaly_features.cpu(), anom_filename)
            # print('save ...')
        
        if True: # (epoch + 1) % 10 == 0:
            #logger.info(f"Epoch [{epoch+1}/{epochs}], Total Loss: {loss.item():.6f}")
            
            log_message = (
                f"Epoch [{epoch+1}/{epochs}], "
                f"Total Loss: {loss.item():.6f} | "
                f"Margin_cls: {loss_margin_cls.item():.6f} | "
                f"Margin_patch: {loss_margin_patch.item():.6f} | "
                f"V2T_Cls: {loss_v2t_cls.item():.6f} | "
                f"V2T_Patch: {loss_v2t_patch.item():.6f} | "
                f"Trip_Cls: {trip_loss_cls.item():.6f} | "
                f"Trip_Patch: {trip_loss_patch.item():.6f} | "
                f"Match_Pos: {loss_match_pos.item():.6f} | "
                f"Match_Neg: {loss_match_neg.item():.6f} "
            )

            logger.info(log_message)
            # 同时确认文件已保存
            # logger.info(f"Saved example embeddings: {norm_filename}, {anom_filename}")

    logger.info("--- Training Finished ---")

    return avg_positive_features, avg_negative_features


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
    dataset_name = obj_name = args.dataset
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
    epochs = args.epochs
    num_prompt_tokens = args.num_prompt_tokens
    margin = args.margin / 10
    reg_lambda = 1 / args.reg_lambda
    print('****** margin ', margin, ' -- reg_lambda ', reg_lambda)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    txt_path = os.path.join(save_path, 'log_train.txt')
    logger = set_logger(txt_path, mode='w')

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

    # obj_name = 'casting_billet'
    '''
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
        obj_list = datasets.CLSNAMES
        test_data = datasets.MetalDataset(root=dataset_dir, meta_path=meta_path, transform=preprocess, target_transform=transform, mode='test', k_shot=k_shot, save_dir=save_path, obj_name=obj_list)
        # obj_list = test_data.get_cls_names()
        print('******* running ... obj_list ', obj_list)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=False)  # 32
    '''

    model.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []  # image level text_probs
    results['pr_sp'] = [] # image level label

    with open(des_path) as f:
        des=json.load(f)
    
    ########################################

    #with torch.no_grad(): 
        # Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features, Mem_redundant_features = prepare_text_feature(model, [obj_name], des_path, use_detailed)

    print('############  k_shot ', args.k_shot, ' obj_name ', obj_name)
    samples = des[obj_name]['samples'][:k_shot]
    print('normal samples ', len(samples))

    print("\nLoading training images...")
    normal_train_images = [ Image.open(x).convert("RGB") for x in samples ]
    logger.info(f"Loaded {len(normal_train_images)} real normal images.")

    # 实例化模型时，传入参考图像
    adapter_model = DualSoftPromptLearner(
        model=model, 
        des_path=des_path, 
        obj_name=obj_name, 
        num_prompt_tokens=num_prompt_tokens, use_detailed=use_detailed
    ).to(device)

    train_adapter(adapter_model, preprocess, normal_train_images, img_size, save_path, logger, epochs, feature_list, dpam_layer, ignore_residual, device, margin=margin, reg_lambda=reg_lambda)



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
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    parser.add_argument("--epochs", type=int, default=10, help="max epoch")
    parser.add_argument("--margin", type=int, default=0, help="margin for loss")
    parser.add_argument("--reg_lambda", type=int, default=1, help="lambda for reg")
    parser.add_argument("--num_prompt_tokens", type=int, default=2, help="num_prompt_tokens")

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
