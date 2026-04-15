import torch
import os, sys
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(os.path.join(o_path, '../'))
from dataset import datasets
import json



from collections import OrderedDict
def initialize_memory(obj_list):

    mid = []
    large = []
    patch = []
    for x in obj_list:
        mid.append((x, []))
        large.append((x, []))
        patch.append((x, []))
    mid_memory   = OrderedDict(mid)
    large_memory = OrderedDict(large)
    patch_memory = OrderedDict(patch)
    return mid_memory, large_memory, patch_memory

from PIL import Image


@torch.no_grad()
def memory_radio(model, obj_list, preprocess, device, feature_list):
    
    des_path = '/data/datasets/pub/public/own_anomaly_detect/datasets_des_info.json'
    with open(des_path) as f:
        result=json.load(f)
    
    miss_obj = set()
    for obj in obj_list:
        samples = result[obj]['samples']
        if len(samples) < 1:
            miss_obj.add(obj)
            continue
    print('miss_obj ', miss_obj)
    obj_list = [x for x in obj_list if x not in miss_obj]

    mem_features = {}
    for obj in obj_list:
        samples = result[obj]['samples']
        features = []

        for image in samples:
            img = Image.open(os.path.join(image))
		    # transforms
            if preprocess is not None:
                img = preprocess(img)
            images = img.to(device).unsqueeze(0)
            cls_name = [obj]
            with torch.no_grad():
                class_tokens, _, patch_tokens = model.encode_image(images, feature_list)
                #patch_tokens = [p[:, 1:, :] for p in patch_tokens]
                features.append(patch_tokens)
        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
    return mem_features


@torch.no_grad()
def memory_surgery(model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot, few_shot_features,
           dataset_name, device, feature_list, dpam_layer):
    
    des_path = '/data/datasets/pub/public/own_anomaly_detect/datasets_des_info.json'
    with open(des_path) as f:
        result=json.load(f)
    
    miss_obj = set()
    for obj in obj_list:
        samples = result[obj]['samples']
        if len(samples) < 1:
            miss_obj.add(obj)
            continue
    print('miss_obj ', miss_obj)
    obj_list = [x for x in obj_list if x not in miss_obj]

    mem_features = {}
    for obj in obj_list:
        samples = result[obj]['samples']
        features = []

        for image in samples:
            img = Image.open(os.path.join(image))
		    # transforms
            if preprocess is not None:
                img = preprocess(img)
            images = img.to(device).unsqueeze(0)
            cls_name = [obj]
            with torch.no_grad():
                class_tokens, _, patch_tokens = model.encode_image(images, feature_list, dpam_layer)
                patch_tokens = [p[:, 1:, :] for p in patch_tokens]
                features.append(patch_tokens)
        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
    return mem_features


@torch.no_grad()
def memory(model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot, few_shot_features,
           dataset_name, device, patch_size):
    
    des_path = './data/text_description/datasets_des_info_ori.json'
    with open(des_path) as f:
        result=json.load(f)
    
    miss_obj = set()
    for obj in obj_list:
        samples = result[obj]['samples']
        if len(samples) < 1:
            miss_obj.add(obj)
            continue
    print('miss_obj ', miss_obj)
    obj_list = [x for x in obj_list if x not in miss_obj]
    mid_memory, large_memory, patch_memory = initialize_memory(obj_list)

    for obj in obj_list:
        samples = result[obj]['samples'][:k_shot]
        print('good samples ', len(samples))
        
        for image in samples:
            img = Image.open(os.path.join(image))
		    # transforms
            if preprocess is not None:
                img = preprocess(img)
            images = img.to(device).unsqueeze(0)
            cls_name = [obj]

            # print("class_name", cls_name)
            large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = model.encode_image(images, patch_size)
            # print("large_scale_tokens", large_scale_tokens.shape, mid_scale_tokens.shape, patch_tokens.shape)
            for class_name, tokens in zip(cls_name, large_scale_tokens):
                large_memory[class_name].append(tokens)
            for class_name, tokens in zip(cls_name, mid_scale_tokens):
                mid_memory[class_name].append(tokens)
            for class_name, tokens in zip(cls_name, patch_tokens):
                patch_memory[class_name].append(tokens)

    for class_name in obj_list:
        large_memory[class_name] = torch.cat(large_memory[class_name])
        mid_memory[class_name] = torch.cat(mid_memory[class_name])
        patch_memory[class_name] = torch.cat(patch_memory[class_name])
        #print("lennnnnshape class_name ", class_name, ' patch_memory shape: ', patch_memory[class_name].shape, ' large_memory shape ', large_memory[class_name].shape, ' mid_memory shape ', mid_memory[class_name].shape)

    return large_memory, mid_memory, patch_memory, miss_obj


@torch.no_grad()
def memory_ori(model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot, few_shot_features,
           dataset_name, device):
    normal_features_ls = {}
    mid_memory, large_memory, patch_memory = initialize_memory(obj_list)
    for i in range(len(obj_list)):
        if dataset_name == 'mvtec':
            normal_data = datasets.MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                       aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path,
                                       obj_name=obj_list[i])
        elif dataset_name == 'visa':
            normal_data = datasets.VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                      mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj_list[i])

        normal_dataloader = torch.utils.data.DataLoader(normal_data, batch_size=1, shuffle=False)
        for index, items in enumerate(normal_dataloader):
            images = items['img'].to(device)
            cls_name = items['cls_name']
            cls_id = items['cls_id']
            patch_size = 16
            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            # print("class_name", cls_name)
            large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = model.encode_image(images, patch_size)
            # print("large_scale_tokens", large_scale_tokens.shape, mid_scale_tokens.shape, patch_tokens.shape)
            for class_name, tokens in zip(cls_name, large_scale_tokens):
                large_memory[class_name].append(tokens)
            for class_name, tokens in zip(cls_name, mid_scale_tokens):
                mid_memory[class_name].append(tokens)
            for class_name, tokens in zip(cls_name, patch_tokens):
                patch_memory[class_name].append(tokens)
            #     print("lennnnnshape", tokens.shape)
            # print("large_memory", large_memory)
            # print("mid_memory", mid_memory)
            # print("large_memory", patch_memory)
    for class_name in obj_list:
        large_memory[class_name] = torch.cat(large_memory[class_name])
        mid_memory[class_name] = torch.cat(mid_memory[class_name])
        patch_memory[class_name] = torch.cat(patch_memory[class_name])
        # print("lennnnnshape", patch_memory[class_name].shape)


    return large_memory, mid_memory, patch_memory