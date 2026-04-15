import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
import json


# April GAN
# 应用于自己的数据集  metal_own
metal_own_map = {'al_224_light':'aluminum sheet metal surface',
                 'casting_billet': 'casting billet metal surface',
                 'steel_pipe': 'steel pipe metal surface',
                 'KolektorSDD': 'kolektor surface',
                 'KolektorSDD2': 'kolektor surface'}


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_prompts = {}
    for obj in objs:
        obj_str = metal_own_map.get(obj, obj)   # metal_own_map
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj_str) for state in prompt_state[i]]
            print(prompted_state[:2])
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = tokenizer(prompted_sentence).to(device)
            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device)
        text_prompts[obj] = text_features

    return text_prompts


class prompt_order():  # prompt 函数
    def __init__(self, des_path) -> None:
        super().__init__()

        with open(des_path) as f:
            des=json.load(f)
        self.total_des = des

        self.state_normal_list_old = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
        ]

        self.state_anomaly_list_old = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage",
        ]

        # April GAN
        self.state_normal_list = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
        # self.state_normal_list = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}']
        self.state_anomaly_list = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

        self.template_list_old =[
            "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a close-up photo of the {}.",
            "a bright photo of a {}.",
            "a bright photo of the {}.",
            "a dark photo of the {}.",
            "a dark photo of a {}.",
            "a jpeg corrupted photo of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a blurry photo of the {}.",
            "a blurry photo of a {}.",
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a small {}.",
            "a photo of the small {}.",
            "a photo of a large {}.",
            "a photo of the large {}.",
            "a photo of the {} for visual inspection.",
            "a photo of a {} for visual inspection.",
            "a photo of the {} for anomaly detection.",
            "a photo of a {} for anomaly detection."
        ]

        # April GAN
        #self.template_list = ['a cropped photo of the {}.']
        self.template_list = ['a bad photo of a {}.', 
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
        # self.template_list = ['{}']


    def prompt(self, class_name, use_detailed=True):
        des_info = self.total_des[class_name]
        print('class_name ', class_name)
        class_name = des_info['map']
        # class_name = 'object'  # 'metal surface'
        print('map class_name ', class_name)

        state_normal_list = self.state_normal_list.copy()
        state_anomaly_list = self.state_anomaly_list.copy()
        
        if use_detailed:
            state_normal_list.extend(des_info['des']['good'])
            state_anomaly_list.extend(des_info['des']['defect'])
        
        print('state_normal_list ', state_normal_list)
        print('state_anomaly_list ', state_anomaly_list)

        class_state = [ele.format(class_name) for ele in state_normal_list]
        normal_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
    
        class_state = [ele.format(class_name) for ele in state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
        print('prompt normal_ensemble_template ', len(normal_ensemble_template), ' anomaly_ensemble_template ', len(anomaly_ensemble_template))

        empty_template = [class_template.format('') for class_template in self.template_list]
        return normal_ensemble_template, anomaly_ensemble_template, empty_template


@torch.no_grad()
def prepare_text_feature(model, obj_list, des_path, use_detailed=True, cache_path='', cache=False):  # 准备文本特征   obj_list: cate list

    cache_file = ''
    if len(cache_path) > 0:
        os.makedirs(cache_path, exist_ok=True)

    Mermory_avg_normal_text_features = []
    Mermory_avg_abnormal_text_features = []
    Mem_redundant_features = []
    text_generator = prompt_order(des_path)

    for obj_name in obj_list:
        
        # 构造缓存文件名
        suffix = '_KeAD' if use_detailed else ''
        cache_file = os.path.join(cache_path, f"{obj_name}_text_features"+suffix+".pt")

        if cache and os.path.exists(cache_file):
            # 从缓存加载
            data = torch.load(cache_file, map_location="cpu")
            normal_text_features = data["normal"]
            abnormal_text_features = data["abnormal"]
            print(f"[Cached] Loaded text features for '{obj_name}' from disk.")
        else:         
            normal_description, abnormal_description, empty_template = text_generator.prompt(obj_name, use_detailed=use_detailed)  # alu
            print('normal_description ', len(normal_description), ' abnormal_description ', len(abnormal_description))
            #normal_description = ['good']
            #abnormal_description = ['damaged']
            '''
            normal_text_features = None
            for x in normal_description:
                if normal_text_features is None:
                    normal_text_features = model.encode_text([x]).float()
                else:
                    normal_text_features += model.encode_text([x]).float()
            normal_text_features /= len(normal_description)
            normal_text_features /= normal_text_features.norm()

            abnormal_text_features = None
            for x in abnormal_description:
                if abnormal_text_features is None:
                    abnormal_text_features = model.encode_text([x]).float()
                else:
                    abnormal_text_features += model.encode_text([x]).float()
            abnormal_text_features /= len(normal_description)
            abnormal_text_features /= abnormal_text_features.norm()
            '''
            normal_text_features = model.encode_text(normal_description).float()
            abnormal_text_features = model.encode_text(abnormal_description).float()
            # empty_features = model.encode_text(empty_template).float()

            normal_text_features = torch.mean(normal_text_features, dim = 0, keepdim= True) 
            normal_text_features /= normal_text_features.norm()
            abnormal_text_features = torch.mean(abnormal_text_features, dim = 0, keepdim= True)   # 取平均
            abnormal_text_features /= abnormal_text_features.norm()
            # redundant_features = torch.mean(empty_features, dim = 0, keepdim= True)   # 取平均
            # redundant_features /= redundant_features.norm()
            # 保存到缓存
            # torch.save({"normal": normal_text_features, "abnormal": abnormal_text_features}, cache_file)
            # print(f"[Saved] Cached text features for '{obj_name}' to disk.")

        Mermory_avg_normal_text_features.append(normal_text_features)
        Mermory_avg_abnormal_text_features.append(abnormal_text_features)
        # Mem_redundant_features.append(redundant_features)
    
    Mermory_avg_normal_text_features = torch.stack(Mermory_avg_normal_text_features)      # [2, 1, 640]  
    Mermory_avg_abnormal_text_features = torch.stack(Mermory_avg_abnormal_text_features)  # [2, 1, 640]  
    # Mem_redundant_features = torch.stack(Mem_redundant_features)
    print('Mermory_avg_normal_text_features ', Mermory_avg_normal_text_features.shape) 
    print('Mermory_avg_abnormal_text_features ', Mermory_avg_abnormal_text_features.shape)   
    # print('Mem_redundant_features ', Mem_redundant_features.shape)

    return Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features, Mem_redundant_features
