#!/bin/bash
CUDA_LAUNCH_BLOCKING=1
few_shots=(0 1 2 4)

for few_num in "${!few_shots[@]}";do

    base_name=KeAD
    des_path=./data/text_description/visa_descriptions_with_samples.json
    meta_path=./data/eval_dataset/VisA_meta.json
    surgery_type=vv_res
    dataset_name=visa
    data_path=/data/datasets/pub/public/own_anomaly_detect/Visa

    save_dir=./output/exps_${base_name}/${dataset_name}_vit_large_14_518_few_shot_${few_shots[few_num]}_${surgery_type}_t1_gpt5_dino_batch_self_update/

    CUDA_VISIBLE_DEVICES=1 python -u main/get_anomaly_map_base.py --dataset ${dataset_name} \
    --save_path ${save_dir} --data_path ${data_path} \
    --des_path ${des_path} --meta_path ${meta_path} \
    --model ViT-L-14-336 --pretrained openai --k_shot ${few_shots[few_num]} \
    --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --update_topk 10 --batch_sim_topk 20 --self_sim_topk 10 --score_topk 10 \
    --surgery_type ${surgery_type}  --use_detailed  --visualize
    wait
done
