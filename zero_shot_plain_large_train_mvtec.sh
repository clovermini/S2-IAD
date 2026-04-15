#!/bin/bash
CUDA_LAUNCH_BLOCKING=1

# 1. 在这里定义你所有想跑的数据集
datasets=("screw")

# 2. 定义你的 few-shot 列表
few_shots=(1)
max_epoch=10

# 3. 外层循环：遍历每一个数据集
for dataset in "${datasets[@]}"; do

    # 4. 内层循环：遍历每一种 k-shot
    for few_num in "${!few_shots[@]}"; do
    
        k_shot_value=${few_shots[few_num]}

        base_name=KeAD_training
        des_path=/data/datasets/pub/public/MVTec_AD/mvtec_descriptions_with_samples.json
        meta_path=/data/datasets/pub/public/MVTec_AD/meta.json
        surgery_type=vv_res
        data_path=/data/datasets/pub/public/MVTec_AD

        # 5. 动态设置 save_dir，使用 $dataset 和 $k_shot_value 变量
        save_dir=./output/exps_${base_name}/${dataset}_vit_large_14_518_few_shot_${k_shot_value}_${surgery_type}_training_dual_soft/

        # 增加日志，方便你跟踪进度
        echo "################################################################"
        echo "## Starting Run:"
        echo "##   Dataset: $dataset"
        echo "##   K-Shot:  $k_shot_value"
        echo "##   Save Dir: $save_dir"
        echo "################################################################"

        # 6. 运行训练脚本，使用 $dataset 和 $k_shot_value 变量
        CUDA_VISIBLE_DEVICES=1 python -u main/get_anomaly_map_base_training.py --dataset ${dataset} \
        --save_path ${save_dir} --data_path ${data_path} --epochs ${max_epoch} \
        --des_path ${des_path} --meta_path ${meta_path} \
        --model ViT-L-14-336 --pretrained openai --k_shot ${k_shot_value} \
        --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --margin 0 --reg_lambda 1000 \
        --surgery_type ${surgery_type}  --use_detailed
        wait

        # 7. 运行测试脚本，使用 $dataset 和 $k_shot_value 变量
        CUDA_VISIBLE_DEVICES=1 python -u main/get_anomaly_map_base_test.py --dataset ${dataset} \
        --save_path ${save_dir} --data_path ${data_path} --load_epoch 10 \
        --des_path ${des_path} --meta_path ${meta_path} \
        --model ViT-L-14-336 --pretrained openai --k_shot ${k_shot_value} \
        --image_size 518 --patch_size 14 --feature_list 6 12 18 24 --dpam_layer 20 --update_topk 20 --batch_sim_topk 50 --self_sim_topk 10 --score_topk 20 \
        --surgery_type ${surgery_type} --use_detailed
        wait

    done # 内层 k-shot 循环结束

done # 外层 dataset 循环结束

echo "All dataset runs finished."