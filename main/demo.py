'''
20231023 chuniliu
各个图像分割类大模型在 各个数据上的预测情况对比
支持不同的模型，不同的数据集
输出预测结果，生成目标检测和图像分割结果，可视化显示
支持不同结果的metric 指标对比
统计不同的模型推理和微调需要的显存情况

1、全自动zero-shot寻找缺陷
2、通过文本描述，寻找缺陷
3、通过少量手工交互寻找效果
4、微调效果，微调encoder、微调decoder、微调prompt

支持优化后的模型对比效果
初步想法：sam基础上，修改prompt结果，微调prompt encoder结果。
patchcore基础上使用大模型的encoder embedding呢？ -- 有效的初筛模型

无人机： ['Discoloration', 'Blistering', 'Cracking', 'Peeling', 'Rust']  ['变色', '起泡', '开裂', '剥落', '生锈']
'''

'''
todo: https://github.com/shenyunhang/APE/tree/main  大模型分割一切
由于fastsam是利用yolov8-seg网络，直接输出所有的instance，不需要grid采样point prompt，，还是fastsam快，fastsam的prompt是一个后处理的过程。sam和mobilesam 的prompt是一个前置输入。默认就是grid sample point
mobilesam
from mobile_encoder.setup_mobile_sam import setup_model
checkpoint = torch.load('../weights/mobile_sam.pt')
mobile_sam = setup_model()
# 加载模型
mobile_sam.load_state_dict(checkpoint,strict=True)


device = "cuda"
mobile_sam.to(device=device)
mobile_sam.eval()
mask_generator = SamAutomaticMaskGenerator(mobile_sam, points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  )# Requires open-cv to run post-processing)        


image = cv2.imread('../notebooks/images/picture1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 推理masks
masks = mask_generator.generate(image) 
'''

import os
import sys
sys.path.append('./')
sys.path.append('../GroundingDINO-main/')   # groundingdino/config/GroundingDINO_SwinB_cfg.py text_encoder_type = "/home/data/liuchuni/projects/fsad_big_model/weights/bert-base-uncased"
# GroundingDINO/groundingdino/util/get_tokenlizer.py if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
sys.path.append('../segment-anything-main/')
sys.path.append('../FastSAM/')   # fastsam/prompt.py 445  clip.load('/home/data/liuchuni/projects/fsad_big_model/weights/clip/ViT-B-32.pt', device=self.device) 
sys.path.append('./Segment-Any-Anomaly/')   # SAA/modelinet.py  113  pretrained_cfg_overlay=dict(file='/mnt/inspurfs/aistation/user-fs/liuchuni/fsad_big_model/weights/wide_resnet50_racm-8234f177.pth')
import tools
import torch
import argparse
import cv2
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import supervision as sv   # /home/data/liuchuni/.conda/envs/chuni_env/lib/python3.9/site-packages/supervision
from supervision.dataset.formats.pascal_voc import detections_to_pascal_voc, load_pascal_voc_annotations, object_to_pascal_voc

# 一些common setting
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
HOME = '/home/data/liuchuni/projects/fsad_big_model/'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE is ', DEVICE)
default_setting = {'dino_config_file':os.path.join(HOME, 'GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
                   'dino_checkpoint':os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth"),
                   'sam_checkpoint':os.path.join(HOME, "weights", "sam_vit_b_01ec64.pth"),
                   'sam_model_version':'vit_b',
                   'fast_sam_checkpoint': os.path.join(HOME, 'weights/FastSAM-s.pt'),
                   'saa_box_threshold': 0.1,
                   'saa_text_threshold': 0.1,
                   'dino_box_threshold': 0.35,
                   'dino_text_threshold': 0.25,
                   'eval_resolution': 1024,
                   'class': ['car', 'dog', 'person', 'nose', 'chair', 'shoe', 'ear']}


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


# 加载不同的模型
def load_model(args, model_type, **kwargs):
    print('kwargs ', kwargs)
    if model_type == 'groundingdino':
        from groundingdino.util.inference import Model
        config_path = kwargs.get('dino_config_file', '')
        checkpoint_path = kwargs.get('dino_checkpoint', '')
        grounding_dino_model = Model(model_config_path=config_path, model_checkpoint_path=checkpoint_path, device=DEVICE)
        return grounding_dino_model
    elif model_type == 'sam':
        from segment_anything import sam_model_registry, SamPredictor
        model_version = kwargs.get('sam_model_version', '')
        checkpoint_path = kwargs.get('sam_checkpoint', '')
        sam = sam_model_registry[model_version](checkpoint=checkpoint_path).to(device=DEVICE)
        sam_predictor = SamPredictor(sam)
        if args.everything:  # 全自动分割一切
            mask_generator = SamAutomaticMaskGenerator(sam)
            return sam_predictor, mask_generator
        return sam_predictor, None
    elif model_type == 'fastsam':
        from fastsam import FastSAM
        checkpoint_path = kwargs.get('fast_sam_checkpoint', '')
        model = FastSAM(checkpoint_path)
        return model
    elif model_type == 'saa':
        import SAA
        # get the model
        model = SAA.Model(
            dino_config_file=kwargs.get('dino_config_file', ''),
            dino_checkpoint=kwargs.get('dino_checkpoint', ''),
            sam_checkpoint=kwargs.get('sam_checkpoint', ''),
            box_threshold=kwargs.get('box_threshold', ''),
            text_threshold=kwargs.get('text_threshold', ''),
            out_size=kwargs.get('eval_resolution', ''),
            device=DEVICE,
        )
        model = model.to(DEVICE)
    else:
        raise NotImplementedError
    

def get_prompt(args, file_name, detections):
    if detections is None:
        if os.path.exists(args.label_path):
            # todo 从xml或者txt中读取到 box坐标
            xml_path = os.path.join(args.label_path, file_name+'.xml')
            xyxy = []
            confidence = []
            image_name, detections, class_names = load_pascal_voc_annotations()  # xml
            #detections = sv.Detections(xyxy=xyxy, confidence=confidence)  # xyxy 应该是xml格式
        elif args.manual:  # 手动交互
            xyxy = []  # 手动交互工具
            detections = sv.Detections(xyxy=xyxy)
            raise NotImplementedError
            # todo  增加point prompt  可以从mask label中采样点
        #else:   # 全自动分割一切
    return detections

    
from fastsam import FastSAMPrompt
from segment_anything import SamAutomaticMaskGenerator
''' text prompt zero-shot || [model: zero_public]'''
# groundingdino+sam
# groundingdino
# fastsam
# groundingdino+fastsam
# saa

''' manual point / box interactive  ||  [model: sam_inter]'''
# sam
# fastsam

''' zero-shot - text prompt / few-shot - memory bank ||  [model:  zero_private / few_private]'''
# winclip
# ...

''' own model ||  [model: zero_own / few_own]'''
#  todo

# 调整为类似于 SAA 的 代码格式， class 
def predict(args):
    timer = tools.Timer()
    timer.start()
    detect_mode = args.mode
    print('detect mode is ', detect_mode)
    if detect_mode in ['zero_public', 'sam_inter']:   # 前提为存在文本，允许输入文本
        if 'groundingdino' in args.model:
            grounding_dino = load_model(args, 'groundingdino', **default_setting)
            timer.stop(update=True, info='load model: groundingdino done!')
        if 'fast_sam' in args.model:
            fast_sam = load_model(args, "fastsam", **default_setting)
            timer.stop(update=True, info='load model: fastsam done!')
        elif 'sam' in args.model:
            sam_predictor, mask_generator = load_model(args, "sam", **default_setting)
            timer.stop(update=True, info='load model: sam done!')
        if 'saa' in args.model:
            saa = load_model(args, 'saa', **default_setting)
    elif detect_mode in ['zero_private', 'zero_own']:  # 文本
        raise NotImplementedError
    elif detect_mode in ['few_private', 'few_own']:  # 参考图，meomory bank
        raise NotImplementedError
    
    data_path = args.data_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_list = []
    if detect_mode in ['zero_public', 'zero_private', 'zero_own']:   # 前提为存在文本，允许输入文本
        text_list = args.text_list
        if len(text_list) == 1:
            text_list = text_list[0]
        if len(text_list) == 0:
            text_list = None
    idx = 0
    for file in os.listdir(data_path):
        if file.strip().split('.')[-1] not in ['JPG', 'jpg', 'jpeg', 'png']:
            continue
        file_name = file.split('.')[0]
        image_path = os.path.join(data_path, file)
        img = cv2.imread(image_path)
        timer.stop(update=True, info='load image '+str(idx)+' : '+file+' done!')
        h, w, d = img.shape
        print('image shape ', img.shape)
        img_list.append(img)
        if detect_mode in ['zero_public', 'zero_private', 'zero_own', 'sam_inter']:   # zero 前提为存在文本，允许输入文本, inter 加入界面交互
            if isinstance(text_list, list):
                text = text_list[idx]
            else:
                text = text_list
            if text is None and 'zero' in detect_mode:
                text = str(input('请为图片'+file+'输入相应文本：'))
                text = text.strip()
            detections = None

            if 'groundingdino' in args.model:
                # detect objects
                detections = grounding_dino.predict_with_classes(  # class 就是 caption
                        image=img,
                        classes=enhance_class_name(class_names=default_setting['class']),
                        box_threshold=default_setting['dino_box_threshold'],
                        text_threshold=default_setting['dino_text_threshold']
                    )
                print('detections ', detections)
                timer.stop(update=True, info='grounding_dino_model detect image done!')

                # annotate image with detections
                box_annotator = sv.BoxAnnotator()
                labels = [
                    f"{default_setting['class'][class_id]} {confidence:0.2f}" 
                    for _, _, confidence, class_id, _ 
                    in detections]
                annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
                # save detect result  xml 格式
                pascal_voc_xml = detections_to_pascal_voc(detections, default_setting['class'], file, img.shape)
                with open(os.path.join(save_path, file_name+'.xml'), "w") as f:
                    f.write(pascal_voc_xml)

                sv.plot_image(annotated_frame, (16, 16))
                timer.stop(update=True, info='grounding_dino_model detect image show done!')

            if 'sam' in args.model and 'fast_sam' not in args.model:  # 依靠于 box/point prompt 或者 界面交互 或者真值？  或者全自动
                sam_predictor.set_image(img)
                timer.stop(update=True, info='sam set image done!')
                result_masks = []
                detections = get_prompt(args, file_name, detections)
                timer.stop(update=True, info='sam get prompt done!')
                
                if detections is None:   # 全自动分割一切  or point
                    if mask_generator is not None:
                        masks = mask_generator.generate(img)
                        detections = sv.Detections(xyxy=[], mask=masks)
                    else:  # todo add point prompt
                        raise NotImplementedError
                else:  # box prompt
                    # box
                    for box in detections.xyxy:
                        masks, scores, logits = sam_predictor.predict(
                                box=box,
                                multimask_output=True
                            )
                        index = np.argmax(scores)
                        result_masks.append(masks[index])
                    detections.mask = np.array(result_masks)
                
                timer.stop(update=True, info='sam segment image done!')

                # annotate image with detections
                mask_annotator = sv.MaskAnnotator()
                annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
                # save masks
                mask = np.zeros((img.shape[0], img.shape[1]))
                for msk in detections.mask:
                    
                    mask += msk
                cv2.imwrite(os.path.join(save_path, file_name+'_sam_mask.png'), mask*255)
                # show box and class
                #annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
                sv.plot_image(annotated_image, (16, 16))
                timer.stop(update=True, info='sam segment image show done!')

            if 'fast_sam' in args.model:  # 分割一切 、 point 、text、boxes prompt
                # 基于yolo的实例分割
                everything_results = fast_sam(image_path, device=DEVICE, retina_masks=True, imgsz=max(h, w), conf=0.4, iou=0.9,)
                timer.stop(update=True, info='fast sam get everything_results done!')
                prompt_process = FastSAMPrompt(image_path, everything_results, device=DEVICE)
                timer.stop(update=True, info='fast sam get prompt_process done!')

                detections = get_prompt(args, file_name, detections)
                timer.stop(update=True, info='fast sam get_prompt done!')

                if detections is None:   # 全自动分割一切  or point
                    if args.everything:
                        # everything prompt
                        ann = prompt_process.everything_prompt()
                    elif text is not None:  # 使用 text prompt
                        # text prompt
                        ann = prompt_process.text_prompt(text=text)
                    else:  # todo add point prompt
                        # point prompt
                        # points default [[0,0]] [[x1,y1],[x2,y2]]
                        # point_label default [0] [1,0] 0:background, 1:foreground
                        #ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])
                        raise NotImplementedError
                else:  # box prompt
                    # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
                    print('fast sam ', detections.xyxy)
                    ann = prompt_process.box_prompt(bboxes=detections.xyxy)
                timer.stop(update=True, info='fast sam prompt_process done!')
                # plot and save
                prompt_process.plot(annotations=ann,output_path=os.path.join(save_path, file_name+'_fastsam_mask.png'),)

            if 'saa' in args.model:  # zero-shot 应用于异常检测的分割
                #textual_prompts = [
                #    ['black melt. dark liquid.', 'capsules'],
                #    ['bubble', 'capsules'],  # 33+-->37+
                #] # detect prompts, filtered phrase
                #property_text_prompts = 'the image of capsule have 20 dissimilar capsule, with a maximum of 1 anomaly. The anomaly would not exceed 1. object area. '

                saa.set_ensemble_text_prompts(text, verbose=False)
                saa.set_property_text_prompts(args.property_text, verbose=False)
                timer.stop(update=True, info='saa set text done!')
                score, appendix = saa(img)
                similarity_map = appendix['similarity_map']
                # save mask
                cv2.imwrite(os.path.join(save_path, file_name+'_saa_mask.png'), score)
                
                plt.subplot(121)
                plt.imshow(img)
                plt.imshow(score, alpha=0.4,cmap='jet')
                plt.title('Anomaly Score')

                plt.subplot(122)
                plt.imshow(img)
                plt.imshow(similarity_map, alpha=0.4, cmap='jet')
                plt.title('Saliency')
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Defect LVLMs Predict", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/uav_bridge", help="path to test dataset")
    parser.add_argument("--label_path", type=str, default="", help="path to test dataset label")
    parser.add_argument("--save_path", type=str, default='./output/uav_bridge', help='path to save results')

    # model
    #parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--mode", type=str, default="zero_public", help="zero_public , zero_private, zero_own, sam_inter")
    parser.add_argument("--model", type=str, default="groundingdino+sam", help="model used")
    parser.add_argument("--text_list", type=str, nargs="+", default=['defect'], help="text list")   # 传入参数之间用空格间隔
    parser.add_argument("--property_text", type=str, default='a image with defect', help="property_text for saa")
    #parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9], help="features used for few shot")
    #parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--everything", type=bool, default=False, help="segmant everything?")
    parser.add_argument("--manual", type=bool, default=False, help="manual interaction?")

    # get and set free gpus
    free_gpu_list = tools.get_free_gpu()
    free_gpus = ','.join(str(x) for x in free_gpu_list[:1])
    print('free_gpus: ', free_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # few shot
    #parser.add_argument("--k_shot", type=int, default=10, help="e.g., 10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    args = parser.parse_args()
    print(args)
    #for key,value in args:
    #    print('args ', key, ' : ', value)

    tools.setup_seed(args.seed)
    predict(args)




