#!/bin/bash
CUDA_LAUNCH_BLOCKING=1
few_shots=(4)
max_epoch=10

for few_num in "${!few_shots[@]}";do

    base_name=KeAD_training
    des_path=./data/text_description/datasets_des_info_metal_part.json
    meta_path=./data/eval_dataset/ksdd_meta.json
    surgery_type=vv_res
    dataset_name=KolektorSDD2
    data_path=/data/datasets/pub/public/own_anomaly_detect

    save_dir=./output/exps_${base_name}/${dataset_name}_vit_large_14_518_few_shot_${few_shots[few_num]}_${surgery_type}_training_dual_soft/

    CUDA_VISIBLE_DEVICES=0 python -u main/get_anomaly_map_base_training.py --dataset ${dataset_name} \
    --save_path ${save_dir} --data_path ${data_path} --epochs ${max_epoch} \
    --des_path ${des_path} --meta_path ${meta_path} \
    --model ViT-L-14-336 --pretrained openai --k_shot ${few_shots[few_num]} \
    --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --margin 0 --reg_lambda 1 \
    --surgery_type ${surgery_type}  --use_detailed
    wait

    CUDA_VISIBLE_DEVICES=0 python -u main/get_anomaly_map_base_test.py --dataset ${dataset_name} \
    --save_path ${save_dir} --data_path ${data_path} --load_epoch 10 \
    --des_path ${des_path} --meta_path ${meta_path} \
    --model ViT-L-14-336 --pretrained openai --k_shot ${few_shots[few_num]} \
    --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --update_topk 30 --batch_sim_topk 20 --self_sim_topk 200 --score_topk 20 \
    --surgery_type ${surgery_type}  --use_detailed
    wait
done
