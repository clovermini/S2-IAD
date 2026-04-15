#!/bin/bash
CUDA_LAUNCH_BLOCKING=1
# few_shots=(0 1 2 4)
few_shots=(1)
max_epoch=10

for few_num in "${!few_shots[@]}";do

    base_name=KeAD_training  # winclip_mvtec
    des_path=/data/datasets/pub/public/own_anomaly_detect/datasets_des_info_gpt4o_v2.json
    meta_path=/data/datasets/pub/public/own_anomaly_detect/metal_meta_d4.json  # metal_own_meta
    surgery_type=vv_res
    dataset_name=KolektorSDD2  # steel_pipe  # casting_billet  #KolektorSDD2  #metal_own
    data_path=/data/datasets/pub/public/own_anomaly_detect

    #save_dir=./output/exps_${base_name}/${dataset_name}_vit_base_16_240_few_shot_${few_shots[few_num]}_${surgery_type}/
    save_dir=./output/exps_${base_name}/${dataset_name}_vit_large_14_518_few_shot_${few_shots[few_num]}_${surgery_type}_training_dual_soft/  # _dino_batch_self_update

    #save_dir=./output/exps_${base_name}/mvtecvit_huge_14_378_few_shot_${few_shots[few_num]}/

    CUDA_VISIBLE_DEVICES=0 python -u main/get_anomaly_map_base_training.py --dataset ${dataset_name} \
    --save_path ${save_dir} --data_path ${data_path} --epochs ${max_epoch} \
    --des_path ${des_path} --meta_path ${meta_path} \
    --model ViT-L-14-336 --pretrained openai --k_shot ${few_shots[few_num]} \
    --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --margin 0 --reg_lambda 1 \
    --surgery_type ${surgery_type}  --use_detailed
    wait


    #save_dir=./output/exps_${base_name}/mvtecvit_huge_14_378_few_shot_${few_shots[few_num]}/

    CUDA_VISIBLE_DEVICES=0 python -u main/get_anomaly_map_base_test.py --dataset ${dataset_name} \
    --save_path ${save_dir} --data_path ${data_path} --load_epoch 10 \
    --des_path ${des_path} --meta_path ${meta_path} \
    --model ViT-L-14-336 --pretrained openai --k_shot ${few_shots[few_num]} \
    --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --update_topk 30 --batch_sim_topk 20 --self_sim_topk 200 --score_topk 20 \
    --surgery_type ${surgery_type}  --use_detailed  # --visualize
    wait
done




#--surgery_type vv \
    #--visualize --save_anomaly_map  --feature_list 6 12 18 24 --use_detailed

# --model ViT-H-14-378-quickgelu --pretrained dfn5b --k_shot ${few_shots[few_num]} --image_size 378 --patch_size 14 --feature_list 8 16 24 32 --dpam_layer 26 
#--model ViT-B-16-plus-240 --pretrained laion400m_e32 --k_shot ${few_shots[few_num]} --image_size 240 --patch_size 16 --feature_list 3 6 9 12 --dpam_layer 10 

