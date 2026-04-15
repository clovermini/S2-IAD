#!/bin/bash
CUDA_LAUNCH_BLOCKING=1
# few_shots=(0 1 2 4)
few_shots=(0 1)

for few_num in "${!few_shots[@]}";do

    base_name=KeAD  # winclip_mvtec
    des_path=/data/datasets/pub/public/own_anomaly_detect/Visa/visa_descriptions_with_samples.json
    meta_path=/data/datasets/pub/public/own_anomaly_detect/Visa/meta.json
    surgery_type=vv_res
    dataset_name=visa
    data_path=/data/datasets/pub/public/own_anomaly_detect/Visa

    #save_dir=./output/exps_${base_name}/${dataset_name}_vit_base_16_240_few_shot_${few_shots[few_num]}_${surgery_type}/
    save_dir=./output/exps_${base_name}/${dataset_name}_vit_large_14_518_few_shot_${few_shots[few_num]}_${surgery_type}_t1_gpt5_dino_batch_self_update/  # _t1_gpt5_dino_update

    #save_dir=./output/exps_${base_name}/mvtecvit_huge_14_378_few_shot_${few_shots[few_num]}/

    CUDA_VISIBLE_DEVICES=1 python -u main/get_anomaly_map_base_cp.py --dataset ${dataset_name} \
    --save_path ${save_dir} --data_path ${data_path} \
    --des_path ${des_path} --meta_path ${meta_path} \
    --model ViT-L-14-336 --pretrained openai --k_shot ${few_shots[few_num]} \
    --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --update_topk 10 --batch_sim_topk 20 --self_sim_topk 10 --score_topk 10 \
    --surgery_type ${surgery_type}  --use_detailed  --visualize
    wait
done

#--surgery_type vv \
    #--visualize --save_anomaly_map  --feature_list 6 12 18 24

# --model ViT-H-14-378-quickgelu --pretrained dfn5b --k_shot ${few_shots[few_num]} --image_size 378 --patch_size 14 --feature_list 8 16 24 32 --dpam_layer 26 
#--model ViT-B-16-plus-240 --pretrained laion400m_e32 --k_shot ${few_shots[few_num]} --image_size 240 --patch_size 16 --feature_list 3 6 9 12 --dpam_layer 10 

