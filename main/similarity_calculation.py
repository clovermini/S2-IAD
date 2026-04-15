import torch
import torch.nn.functional as F


def print_tensor_stats(tensor, name="Tensor"):
    """打印一个张量的基本统计信息"""
    print(
        f"--- {name} 统计信息 ---\n"
        f"  数值范围 (Min): {tensor.min().item():.6f}\n"
        f"  数值范围 (Max): {tensor.max().item():.6f}\n"
        f"  均值 (Mean)  : {tensor.mean().item():.6f}\n"
        f"  标准差 (Std) : {tensor.std().item():.6f}\n"
    )

def normalize_map(anomaly_map):
    """将一批异常图分别归一化到 [0, 1]"""
    B, C, H, W = anomaly_map.shape
    map_flat = anomaly_map.view(B, C, -1)
    map_min = map_flat.min(dim=-1, keepdim=True)[0]
    map_max = map_flat.max(dim=-1, keepdim=True)[0]
    map_scaled = torch.where(map_max > map_min, 
                             (map_flat - map_min) / (map_max - map_min), 
                             torch.zeros_like(map_flat))
    return map_scaled.view(B, C, H, W)


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


def get_self_sim(patch_feature):
    '''
    自相似异常图
    patch_feature: torch.Size([5, 1369, 768])
    '''
    B, N, D = patch_feature.shape
    H = W = int(N ** 0.5)

    # 1. L2 归一化
    patch_feature = F.normalize(patch_feature, dim=-1)  # [B, N, D]

    # 2. 计算两两余弦相似度矩阵 (1 - cosθ) -> [N, N]
    sim_matrix = torch.bmm(patch_feature, patch_feature.transpose(1, 2))  # [B, N, N]

    global_sim = sim_matrix.mean(dim=1)  # [B, N] 每个patch与全局的相似度

    global_sim = 1 - global_sim.view(B, H, W)
    
    return global_sim


def get_self_sim_topk_batch(patch_feature, initial_anomaly_map, k_self_sim=5, k_update=5, reduction='mean'):
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

     # 1. 确定需要计算的最大的 k 值，以保证一次 topk 调用就足够
    # 如果不需要更新，则 k_for_topk 就是 k_self_sim
    k_for_topk = k_self_sim
    if initial_anomaly_map is not None:
        k_for_topk = max(k_self_sim, k_update)

    # 4. 对每个 patch，找到 top-k 最近邻的距离和索引
    topk_dist_full, topk_indices_full = torch.topk(dist_matrix, k=k_for_topk, largest=False, dim=-1)  # [B, N, k]

    topk_dist_for_sim = topk_dist_full[:, :, :k_self_sim]

    # 5. 聚合 top-k 距离
    if reduction == 'mean':
        anomaly_score = topk_dist_for_sim.mean(dim=-1)  # [B, N]
    elif reduction == 'min':
        anomaly_score = topk_dist_for_sim.min(dim=-1)[0]  # [B, N]
    else:
        raise ValueError("reduction must be 'mean' or 'min'")
    # 6. reshape 回 (B, H, W) 得到异常得分图
    anomaly_map = anomaly_score.view(B, H, W)  # [B, H, W]

    if initial_anomaly_map is None:
        return anomaly_map, None

    topk_indices_for_update = topk_indices_full[:, :, :k_update]

    # 7. 更新初始异常得分图
    # 将 top-k 分值从 initial_anomaly_map 中提取出来并聚合
    initial_anomaly_map_flat = initial_anomaly_map.view(B, -1)  # [B, N]
    # 检查索引范围
    # Gather top-k scores using advanced indexing
    batch_indices = torch.arange(B, device=patch_feature.device).view(B, 1, 1).expand(B, N, k_update)
    topk_initial_scores = initial_anomaly_map_flat[batch_indices, topk_indices_for_update]  # [B, N, k]
    
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


import torch
import torch.nn.functional as F

def adaptive_fusion_spatial(anomaly_map_list, T=1.0):
    """
    使用像素级的Softmax作为空间注意力，自适应地融合两个异常图。
    
    :param self_sim_map: Tensor [B, H, W], 图内自相似度异常图 (需要先归一化到0-1)
    :param batch_sim_map: Tensor [B, H, W], 图像间相似度异常图 (需要先归一化到0-1)
    :param T: 温度系数，用于调节Softmax的平滑程度。
    :return: fused_map: Tensor [B, H, W], 融合后的异常图。
    """
    # 1. 将两个map堆叠成一个新的维度，形状变为 [B, 2, H, W]
    stacked_maps = torch.stack(anomaly_map_list, dim=1)
    # stacked_maps_norm = normalize_map(stacked_maps) 
    
    # 2. 沿着新创建的维度(dim=1)计算Softmax，得到空间权重图
    # weight_maps shape: [B, 2, H, W]
    weight_maps = F.softmax(stacked_maps / T, dim=1)
    
    # 3. 将权重图与原始图相乘并求和
    # (weight_maps * stacked_maps) 得到加权后的两个图
    # .sum(dim=1) 沿着新维度求和，完成融合
    fused_map = torch.sum(weight_maps * stacked_maps, dim=1)
    
    return fused_map


import torch
import torch.nn.functional as F

# --- Placeholders (Replace with your actual models/logic) ---

def extract_features(patches):
    """Placeholder for a feature extractor (e.g., a pretrained CNN)."""
    B, C, H, W = patches.shape
    # In a real scenario, use a neural network: return feature_extractor_model(patches)
    # For this example, we flatten, which is a basic form of feature extraction.
    return patches.view(B, -1)

def calculate_initial_anomaly_scores(features):
    """Placeholder to get initial anomaly scores from features."""
    # Example: A simple score could be the L2 norm of the feature vector
    scores = torch.linalg.norm(features, dim=1)
    return scores

# --- Main Synthesized Logic ---

# patch14 的异常得分和patch 16的dino feature相似度结合计算update
def non_local_multi_scale_anomaly(image_tensor, k=5):
    """
    Implements the full logic:
    1. Gets initial scores from 14x14 patches.
    2. Maps these scores to the 16x16 grid.
    3. Averages scores based on 16x16 feature similarity.
    """
    if image_tensor.dim() != 4 or image_tensor.shape[0] != 1:
        raise ValueError("Input tensor must have shape (1, C, H, W)")
        
    _, C, H, W = image_tensor.shape
    
    # --- Step 1: Feature Extraction on Both Grids ---
    patches_14 = image_tensor.unfold(2, 14, 14).unfold(3, 14, 14).contiguous().view(-1, C, 14, 14)
    patches_16 = image_tensor.unfold(2, 16, 16).unfold(3, 16, 16).contiguous().view(--1, C, 16, 16)
    
    features_14 = extract_features(patches_14)
    features_16 = extract_features(patches_16)

    # --- Step 2: Calculate Initial Scores on the 14x14 Grid ---
    initial_scores_14 = calculate_initial_anomaly_scores(features_14)
    
    # --- Step 3: Map 14x14 Scores to the 16x16 Grid ---
    # A: Create a fine-grained, pixel-level score map
    H_14, W_14 = H // 14, W // 14
    score_map_14 = initial_scores_14.view(1, 1, H_14, W_14)
    pixel_level_score_map = F.interpolate(score_map_14, size=(H, W), mode='nearest')
    
    # B: Aggregate the pixel map down to the 16x16 grid
    # This gives one aggregated score for each 16x16 patch.
    aggregated_scores_16 = F.avg_pool2d(pixel_level_score_map, kernel_size=16, stride=16).squeeze()
    aggregated_scores_16 = aggregated_scores_16.view(-1) # Flatten to a vector

    # --- Step 4: Perform Similarity-Based Averaging on the 16x16 Grid ---
    # A: Find top-k similar patches based on 16x16 features
    features_16_norm = F.normalize(features_16, p=2, dim=1)
    similarity_matrix = torch.matmul(features_16_norm, features_16_norm.t())
    _, topk_indices = torch.topk(similarity_matrix, k + 1, dim=1)
    topk_indices = topk_indices[:, 1:] # Exclude self-similarity

    # B: Gather the aggregated scores of the most similar patches
    num_patches_16 = features_16.shape[0]
    scores_of_similar_patches = torch.gather(
        aggregated_scores_16.unsqueeze(0).expand(num_patches_16, -1), 
        1, 
        topk_indices
    )
    
    # C: Average the scores to get the final, context-aware score
    final_scores_16 = torch.mean(scores_of_similar_patches, dim=1)

    # --- Step 5: Reshape Final Scores into a Coarse Map ---
    H_16, W_16 = H // 16, W // 16
    final_score_map = final_scores_16.view(H_16, W_16)

    return final_score_map