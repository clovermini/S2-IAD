#!/bin/bash
CUDA_LAUNCH_BLOCKING=1
train_few_shots=4
few_shots=(4)

for few_num in "${!few_shots[@]}";do

    base_name=KeAD_training
    des_path=./data/text_description/datasets_des_info_metal_part.json
    meta_path=./data/eval_dataset/metal_meta.json
    surgery_type=vv_res
    dataset_name=casting_billet
    data_path=/data/datasets/pub/public/own_anomaly_detect

    save_dir=./output/exps_${base_name}/${dataset_name}_vit_large_14_518_few_shot_${train_few_shots}_${surgery_type}_training_dual_soft/

    CUDA_VISIBLE_DEVICES=1 python -u main/get_anomaly_map_base_test.py --dataset ${dataset_name} \
    --save_path ${save_dir} --data_path ${data_path} --load_epoch 10 \
    --des_path ${des_path} --meta_path ${meta_path} \
    --model ViT-L-14-336 --pretrained openai --k_shot ${few_shots[few_num]} \
    --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --update_topk 1 --batch_sim_topk 1 --self_sim_topk 1 --score_topk 1 \
    --surgery_type ${surgery_type}  --use_detailed
    wait
done
