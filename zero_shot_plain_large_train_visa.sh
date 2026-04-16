#!/bin/bash
CUDA_LAUNCH_BLOCKING=1

datasets=("pcb2")
few_shots=(4)
max_epoch=10

for dataset in "${datasets[@]}"; do

    for few_num in "${!few_shots[@]}"; do
    
        k_shot_value=${few_shots[few_num]}

        base_name=KeAD_training
        des_path=./data/text_description/visa_descriptions_with_samples.json
        meta_path=./data/eval_dataset/VisA_meta.json
        surgery_type=vv_res
        data_path=/data/datasets/pub/public/own_anomaly_detect/Visa

        save_dir=./output/exps_${base_name}/${dataset}_vit_large_14_518_few_shot_${k_shot_value}_${surgery_type}_training_dual_soft/

        echo "################################################################"
        echo "## Starting Run:"
        echo "##   Dataset: $dataset"
        echo "##   K-Shot:  $k_shot_value"
        echo "##   Save Dir: $save_dir"
        echo "################################################################"

        CUDA_VISIBLE_DEVICES=1 python -u main/get_anomaly_map_base_training.py --dataset ${dataset} \
        --save_path ${save_dir} --data_path ${data_path} --epochs ${max_epoch} \
        --des_path ${des_path} --meta_path ${meta_path} \
        --model ViT-L-14-336 --pretrained openai --k_shot ${k_shot_value} \
        --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --margin 0 --reg_lambda 1 \
        --surgery_type ${surgery_type}  --use_detailed
        wait

        CUDA_VISIBLE_DEVICES=1 python -u main/get_anomaly_map_base_test.py --dataset ${dataset} \
        --save_path ${save_dir} --data_path ${data_path} --load_epoch 10 \
        --des_path ${des_path} --meta_path ${meta_path} \
        --model ViT-L-14-336 --pretrained openai --k_shot ${k_shot_value} \
        --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --update_topk 10 --batch_sim_topk 20 --self_sim_topk 10 --score_topk 10 \
        --surgery_type ${surgery_type} --use_detailed
        wait

    done

done

echo "All dataset runs finished."
