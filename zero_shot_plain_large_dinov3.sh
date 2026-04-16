#!/bin/bash
CUDA_LAUNCH_BLOCKING=1
few_shots=(0 1 2 4)

for few_num in "${!few_shots[@]}";do

    base_name=KeAD
    des_path=./data/text_description/datasets_des_info_metal_part.json
    meta_path=./data/eval_dataset/metal_meta.json
    surgery_type=vv_res
    dataset_name=metal_own
    data_path=/data/datasets/pub/public/own_anomaly_detect

    save_dir=./output/exps_${base_name}/${dataset_name}_vit_large_14_448_few_shot_${few_shots[few_num]}_${surgery_type}_batch_dinov3/

    CUDA_VISIBLE_DEVICES=2 python -u main/get_anomaly_map_base_dinov3.py --dataset ${dataset_name} \
    --save_path ${save_dir} --data_path ${data_path}\
    --des_path ${des_path} --meta_path ${meta_path} \
    --model ViT-L-14-336 --pretrained openai --k_shot ${few_shots[few_num]} \
    --image_size 448 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --batch_sim_topk 24 --self_sim_topk 10 \
    --surgery_type ${surgery_type} --use_detailed
    wait
done
