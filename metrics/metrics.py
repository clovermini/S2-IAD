'''
20250411 chuniliu
Defect-detection evaluation metrics.
Anomaly detection: pAUROC / PRO / F1-max.
Object detection: precision / recall / mAP@0.5 / mAP@.5:.95 / p-r curve / confusion matrix.
Defect detection (multi-class): same as object detection, plus miss rate / false positive rate / accuracy /
semantic segmentation metrics such as pixel accuracy, mean pixel accuracy (per class), and mean IoU.
In practice, object-detection-level accuracy is often sufficient.
Efficiency metrics: Params / GFLOPS / FPS.
'''

import os
import numpy as np
from PIL import Image
from tabulate import tabulate
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):  # Compute PRO-AUC.
    """Compute the PRO-AUC metric for pixel-level anomaly maps."""
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def cal_metrics(obj_list, results):
    """Compute per-class and mean anomaly-detection metrics."""
    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        pr_sp_tmp = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
                pr_sp_tmp.append(np.max(results['anomaly_maps'][idxes]))
                gt_sp.append(results['gt_sp'][idxes])
                pr_sp.append(results['pr_sp'][idxes])
        gt_px = np.array(gt_px)
        gt_sp = np.array(gt_sp)
        pr_px = np.array(pr_px)
        pr_sp = np.array(pr_sp)

        # print('gt_px ', gt_px.shape, ' gt_px ', np.unique(gt_px), ' obj ', obj)
        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        '''  # other metrics
        ap_sp = average_precision_score(gt_sp, pr_sp)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        # print("precisions recalls", precisions, recalls)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        # print("precisions recalls", precisions, recalls)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro = cal_pro_score(gt_px, pr_px)
        '''
        f1_px = ap_px = aupro = f1_sp = ap_sp = 0.0

        table.append(str(np.round(auroc_px * 100, decimals=1)))
        table.append(str(np.round(f1_px * 100, decimals=1)))
        table.append(str(np.round(ap_px * 100, decimals=1)))
        table.append(str(np.round(aupro * 100, decimals=1)))
        table.append(str(np.round(auroc_sp * 100, decimals=1)))
        table.append(str(np.round(f1_sp * 100, decimals=1)))
        table.append(str(np.round(ap_sp * 100, decimals=1)))

        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_ls.append(aupro)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)


    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_px_ls) * 100, decimals=1)), str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(aupro_ls) * 100, decimals=1)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_sp_ls) * 100, decimals=1)), str(np.round(np.mean(ap_sp_ls) * 100, decimals=1))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
                                          'f1_sp', 'ap_sp'], tablefmt="pipe")
    return results


from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def cal_pro_score_optimized(masks, amaps, max_step=200, expect_fpr=0.3):
    """Optimized PRO-AUC computation with more vectorized operations."""
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    
    # Pre-calculate inverse masks once
    inverse_masks = 1 - masks
    
    for th in np.arange(min_th, max_th, delta):
        # Vectorized thresholding
        binary_amaps = amaps > th
        
        # Vectorized PRO calculation
        labeled_masks = np.array([measure.label(mask) for mask in masks])
        pro = []
        for i in range(len(masks)):
            regions = measure.regionprops(labeled_masks[i])
            for region in regions:
                coords = region.coords
                tp_pixels = binary_amaps[i][coords[:, 0], coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        
        # Vectorized FPR calculation
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        
        pros.append(np.mean(pro) if pro else 0)
        fprs.append(fpr)
        ths.append(th)
    
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def cal_metrics_optimized(obj_list, results, compute_all=False):
    """Compute metrics with parallel per-class processing."""
    # Precompute all necessary data outside loops
    cls_names = results['cls_names']
    imgs_masks = results['imgs_masks'].squeeze(1).numpy() if len(results['imgs_masks'].shape) == 4 else results['imgs_masks'].numpy()
    anomaly_maps = results['anomaly_maps'].squeeze(1) if len(results['anomaly_maps'].shape) == 4 else results['anomaly_maps']
    gt_sp = results['gt_sp']
    pr_sp = results['pr_sp']
    
    # Process each object in parallel
    def process_object(obj):
        # Get indices for current object
        idxes = np.where(cls_names == obj)[0].tolist()
        
        # Extract relevant data using list indexing
        obj_gt_px = np.stack([imgs_masks[i] for i in idxes])
        obj_pr_px = np.stack([anomaly_maps[i] for i in idxes])
        obj_gt_sp = gt_sp[idxes]
        obj_pr_sp = pr_sp[idxes]
        
        # Compute metrics
        metrics = {
            'object': obj,
            'auroc_px': roc_auc_score(obj_gt_px.ravel(), obj_pr_px.ravel()),
            'auroc_sp': roc_auc_score(obj_gt_sp, obj_pr_sp)
        }
        
        if compute_all:
            metrics.update({
                'ap_sp': average_precision_score(obj_gt_sp, obj_pr_sp),
                'ap_px': average_precision_score(obj_gt_px.ravel(), obj_pr_px.ravel()),
                'f1_sp': compute_f1_score(obj_gt_sp, obj_pr_sp),
                'f1_px': compute_f1_score(obj_gt_px.ravel(), obj_pr_px.ravel()),
                'aupro': cal_pro_score_optimized(obj_gt_px, obj_pr_px) if len(obj_gt_px.shape) == 2 else cal_pro_score(obj_gt_px.squeeze(1), obj_pr_px.squeeze(1))
            })
        else:
            metrics.update({
                'f1_px': 0.0, 'ap_px': 0.0, 'aupro': 0.0,
                'f1_sp': 0.0, 'ap_sp': 0.0
            })
        
        return metrics
    
    # Parallel processing
    all_metrics = Parallel(n_jobs=-1)(delayed(process_object)(obj) for obj in obj_list)
    
    # Prepare results table
    table_ls = []
    metric_sums = {k: [] for k in all_metrics[0].keys() if k != 'object'}
    
    for metrics in all_metrics:
        row = [metrics['object']] + [str(np.round(metrics[k] * 100, decimals=1)) for k in metric_sums]
        table_ls.append(row)
        for k in metric_sums:
            metric_sums[k].append(metrics[k])
    
    # Add mean row
    mean_row = ['mean'] + [str(np.round(np.mean(vals) * 100, decimals=1)) for vals in metric_sums.values()]
    table_ls.append(mean_row)
    
    # Generate final table
    headers = ['objects'] + list(metric_sums.keys())
    return tabulate(table_ls, headers=headers, tablefmt="pipe")

def compute_f1_score(y_true, y_score):
    """Return the maximum F1 score over the precision-recall curve."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
    return np.max(f1_scores[np.isfinite(f1_scores)])
