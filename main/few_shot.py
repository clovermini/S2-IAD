"""Utilities for building few-shot visual memory from reference images.

This module loads a small number of normal samples for each class, encodes
them with the current vision backbone, and stores the resulting features for
later retrieval during anomaly scoring.
"""

import torch
import os, sys
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(os.path.join(o_path, '../'))

from PIL import Image
import json
from collections import OrderedDict


def initialize_memory(obj_list):
    """Create ordered memory containers for each object category."""

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



@torch.no_grad()
def memory_surgery(model, obj_list, des_path, preprocess, k_shot, device, feature_list, dpam_layer, ignore_residual=False):
    """Build few-shot memory features from the sampled normal references.

    Args:
        model: Vision-language model wrapper that exposes image encoders.
        obj_list: Categories that should contribute few-shot references.
        des_path: JSON file containing sample image paths per category.
        preprocess: Image preprocessing pipeline used by the current model.
        k_shot: Number of reference samples to keep for each category.
        device: Target device for feature extraction.
        feature_list: Reserved layer selection argument for image features.
        dpam_layer: Reserved DPAM layer argument for image features.
        ignore_residual: Reserved flag for residual handling.

    Returns:
        A dictionary that maps each class name to a list of encoded features.
    """
    
    with open(des_path) as f:
        des=json.load(f)
    
    miss_obj = set()
    for obj in obj_list:
        samples = des[obj]['samples']
        if len(samples) < 1:
            miss_obj.add(obj)
            continue
    print('miss_obj ', miss_obj)
    obj_list = [x for x in obj_list if x not in miss_obj]

    # Encode the first k normal samples of every valid class and keep them in memory.
    mem_features = {}
    for obj in obj_list:
        samples = des[obj]['samples'][:k_shot]
        print('good samples ', len(samples))

        features = []

        for image in samples:
            img = Image.open(os.path.join(image))
		    # transforms
            if preprocess is not None:
                img = preprocess(img)
            images = img.to(device).unsqueeze(0)
            cls_name = [obj]
            with torch.no_grad():
                # use clip encoder features
                #class_tokens, _, patch_tokens = model.encode_image(images, feature_list, dpam_layer, ignore_residual)
                #patch_tokens = [p[:, 1:, :] for p in patch_tokens]
                #features.append(patch_tokens)

                # use dino encoder features
                dino_features = model.get_dino_features(images)
                features.append(dino_features)
                
        # Concatenate feature tensors from all selected support images layer by layer.
        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
        # print('mem_features [obj ] ', len(mem_features[obj]), ' -- ', mem_features[obj][0].size())
    return mem_features
