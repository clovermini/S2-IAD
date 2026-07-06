# S2-IAD

Official PyTorch implementation of **S2-IAD: Constrained Semantic Adaptation and Structural Evidence Calibration for Industrial Anomaly Detection**.

This repository implements an industrial anomaly detection pipeline built on top of:

- CLIP-based semantic matching
- structural anomaly modeling from patch similarity
- few-shot memory built from normal reference images
- prompt learning for dataset-specific text adaptation

The codebase currently supports zero-shot and few-shot experiments on:

- MVTec AD
- VisA
- KolektorSDD
- KolektorSDD2
- custom metal anomaly datasets, including `casting_billet`, `steel_pipe`

For the metal datasets used in this repository, `casting_billet` and `steel_pipe`
are sourced from the public **MVIT_metal_datasets** repository:
https://github.com/clovermini/MVIT_metal_datasets

## Overview

The project combines three sources of anomaly evidence:

- **semantic text-image matching** from CLIP prompts
- **structural patch similarity** from DINO/DINOv3 features
- **few-shot normal memory** constructed from support images

For few-shot training, the repository also includes a prompt-learning stage that adjusts text features with normal reference images and then reuses the learned text embeddings during testing.

## Repository Structure

```text
S2-IAD/
├── data/
│   ├── eval_dataset/          # meta files for evaluation splits
│   ├── preprocess/            # scripts for metadata generation and data conversion
│   └── text_description/      # text descriptions and prompt metadata
├── dataset/
│   └── datasets.py            # dataset loader for metal datasets
├── main/
│   ├── get_anomaly_map_base.py
│   ├── get_anomaly_map_base_test.py
│   ├── get_anomaly_map_base_training.py
│   ├── get_anomaly_map_base_dinov3.py
│   ├── few_shot.py
│   ├── prompt_ensemble.py
│   └── similarity_calculation.py
├── metrics/
│   └── metrics.py             # AUROC / AP / PRO and related metrics
├── models/
│   ├── open_clip/
│   ├── dinov2/
│   ├── dinov3-main/
│   └── transformers/
├── utils/
│   ├── tools.py
│   └── visualizer.py
└── zero_shot_plain_*.sh       # example running scripts
```

## Main Entry Points

### Zero-shot / few-shot inference

- `main/get_anomaly_map_base.py`
  Main inference entry for CLIP + DINO-based anomaly detection.

- `main/get_anomaly_map_base_dinov3.py`
  Variant that uses DINOv3 hidden states for structural similarity.

### Few-shot prompt learning

- `main/get_anomaly_map_base_training.py`
  Learns prompt tokens from normal support images and saves adapted text embeddings.

- `main/get_anomaly_map_base_test.py`
  Loads the saved prompt embeddings and evaluates the trained few-shot model.

### Supporting modules

- `main/prompt_ensemble.py`
  Builds prompt templates and text prototypes.

- `main/few_shot.py`
  Builds few-shot visual memory from support samples.

- `main/similarity_calculation.py`
  Patch similarity, self-similarity, batch similarity, and map fusion.

## Data Files in This Repository

This repository already includes local metadata and text-description files:

### Metadata

- `data/eval_dataset/MVTec_meta.json`
- `data/eval_dataset/VisA_meta.json`
- `data/eval_dataset/metal_meta.json`
- `data/eval_dataset/ksdd_meta.json`

### Text descriptions

- `data/text_description/mvtec_descriptions_with_samples.json`
- `data/text_description/visa_descriptions_with_samples.json`
- `data/text_description/datasets_des_info_gpt4o_v2.json`
- `data/text_description/datasets_des_info_metal_part.json`

These files define:

- category names
- image and mask paths
- text prompts / descriptions
- few-shot reference sample paths

### Metal dataset source

The metal subset in this project is based on external public datasets.

- `casting_billet` and `steel_pipe` come from **MVIT_metal_datasets**
  Link: https://github.com/clovermini/MVIT_metal_datasets

According to the MVIT repository, it releases metal surface defect datasets with
pixel-level annotations, including Casting Billet and Steel Pipe.

## Dataset Layout

The code expects the actual image datasets to exist on disk, while metadata and text descriptions are loaded from this repository.

For metal datasets, samples are typically described in an MVTec-style layout:

```text
<dataset_root>/
├── <class_name>/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   └── <defect_type>/
│   └── ground_truth/
│       └── <defect_type>/
```

For MVTec and VisA, the code also supports their standard class-based directory organization.

## Data Preparation

The scripts under `data/preprocess/` help generate metadata or convert raw datasets into the format expected by the main pipeline:

- `data/preprocess/mvtec.py`
- `data/preprocess/visa.py`
- `data/preprocess/visa_old.py`
- `data/preprocess/generate_meta.py`
- `data/preprocess/convert_metal_own.py`

Example use cases:

- build `meta.json`-style files for MVTec / VisA / custom metal datasets
- convert custom metal datasets into a unified directory structure
- create small text-description subsets for specific experiments

## Environment

There is currently no pinned `requirements.txt` in this repository, so dependencies need to be installed manually.

At minimum, the code uses:

- `python >= 3.9`
- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `tqdm`
- `opencv-python`
- `matplotlib`
- `scikit-image`
- `scikit-learn`
- `tabulate`
- `transformers`
- `joblib`
- `setuptools`

A typical setup might look like:

```bash
conda create -n s2iad python=3.9 -y
conda activate s2iad
pip install torch torchvision
pip install numpy pillow tqdm opencv-python matplotlib scikit-image scikit-learn tabulate transformers joblib setuptools
```

## Model Weights

The project relies on multiple pretrained backbones:

- OpenCLIP models loaded through the local `models/open_clip/`
- DINOv2 weights for `main/get_anomaly_map_base.py` and `main/get_anomaly_map_base_test.py`
- DINOv3 weights for `main/get_anomaly_map_base_dinov3.py`

Some scripts assume local pretrained weights exist at paths such as:

- `/XXX/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`
- `/XXX/.cache/torch/hub/checkpoints/dinov3`

You may need to update these paths in the code or provide the weights at the expected locations.

## Quick Start

### 1. Zero-shot inference on metal datasets

```bash
python -u main/get_anomaly_map_base.py \
  --dataset metal_own \
  --data_path /path/to/own_anomaly_detect \
  --des_path ./data/text_description/datasets_des_info_metal_part.json \
  --meta_path ./data/eval_dataset/metal_meta.json \
  --save_path ./output/demo_metal \
  --model ViT-L-14-336 \
  --pretrained openai \
  --k_shot 0 \
  --image_size 518 \
  --patch_size 14 \
  --feature_list 6 12 18 24 \
  --dpam_layer 20 \
  --update_topk 40 \
  --batch_sim_topk 5 \
  --self_sim_topk 100 \
  --score_topk 100 \
  --surgery_type vv_res \
  --use_detailed \
  --visualize
```

### 2. Zero-shot inference on VisA

```bash
python -u main/get_anomaly_map_base.py \
  --dataset visa \
  --data_path /path/to/Visa \
  --des_path ./data/text_description/visa_descriptions_with_samples.json \
  --meta_path ./data/eval_dataset/VisA_meta.json \
  --save_path ./output/demo_visa \
  --model ViT-L-14-336 \
  --pretrained openai \
  --k_shot 0 \
  --image_size 518 \
  --patch_size 14 \
  --feature_list 6 12 18 24 \
  --dpam_layer 20 \
  --update_topk 10 \
  --batch_sim_topk 20 \
  --self_sim_topk 10 \
  --score_topk 10 \
  --surgery_type vv_res \
  --use_detailed \
  --visualize
```

## Few-Shot Prompt Learning

Few-shot training is split into two stages:

### Stage 1: train prompt embeddings

```bash
python -u main/get_anomaly_map_base_training.py \
  --dataset KolektorSDD2 \
  --data_path /path/to/own_anomaly_detect \
  --des_path ./data/text_description/datasets_des_info_metal_part.json \
  --meta_path ./data/eval_dataset/ksdd_meta.json \
  --save_path ./output/train_ksdd2 \
  --model ViT-L-14-336 \
  --pretrained openai \
  --k_shot 4 \
  --epochs 10 \
  --image_size 518 \
  --patch_size 14 \
  --feature_list 6 12 18 24 \
  --dpam_layer 20 \
  --margin 0 \
  --reg_lambda 1 \
  --surgery_type vv_res \
  --use_detailed
```

This stage saves learned text embeddings under:

```text
<save_path>/adjusted_text_features/
```

### Stage 2: evaluate trained prompt embeddings

```bash
python -u main/get_anomaly_map_base_test.py \
  --dataset KolektorSDD2 \
  --data_path /path/to/own_anomaly_detect \
  --des_path ./data/text_description/datasets_des_info_metal_part.json \
  --meta_path ./data/eval_dataset/ksdd_meta.json \
  --save_path ./output/train_ksdd2 \
  --load_epoch 10 \
  --model ViT-L-14-336 \
  --pretrained openai \
  --k_shot 4 \
  --image_size 518 \
  --patch_size 14 \
  --feature_list 6 12 18 24 \
  --dpam_layer 20 \
  --update_topk 30 \
  --batch_sim_topk 20 \
  --self_sim_topk 200 \
  --score_topk 20 \
  --surgery_type vv_res \
  --use_detailed
```

## Running the Provided Shell Scripts

The `zero_shot_plain_*.sh` files are experiment templates used for the authors' runs.

Examples:

- `zero_shot_plain_large.sh`
- `zero_shot_plain_large_visa.sh`
- `zero_shot_plain_large_train.sh`
- `zero_shot_plain_large_test.sh`
- `zero_shot_plain_large_train_mvtec.sh`
- `zero_shot_plain_large_train_visa.sh`

Before running them, you will usually need to update:

- `CUDA_VISIBLE_DEVICES`
- `data_path`
- model-weight paths expected inside Python scripts
- dataset-specific save paths

## Outputs

Typical outputs include:

- anomaly maps
- visualization grids
- log files
- saved prompt embeddings
- metric tables

Examples:

```text
output/
└── exps_KeAD/
    └── ...
        ├── log.txt
        ├── vis/
        ├── anomaly_map/
        └── adjusted_text_features/
```

## Evaluation

Metrics are computed in `metrics/metrics.py`, including:

- image-level AUROC
- pixel-level AUROC
- AP
- PRO / pAUROC utilities
- mean results over categories

Visualizations are generated through `utils/visualizer.py`.

## Notes

- Metadata files in `data/eval_dataset/` are local repository assets, but actual image roots are external and must exist on your machine.
- Some paths in the code are currently hard-coded for the original training environment.
- The root shell scripts are best treated as reproducible experiment templates rather than fully portable launchers.
- The repository contains local copies of several model libraries under `models/`, so behavior may differ from upstream releases.

## Citation

If you use this repository in your research, please cite the S2-IAD paper once the official bibliographic information is available.

## License

This project is released under the license in [LICENSE](./LICENSE).
