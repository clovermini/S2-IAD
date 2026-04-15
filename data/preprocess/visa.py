"""Generate a `meta.json` file for the VisA dataset in folder-based format.

This version expects VisA data to be arranged in class/train/test/ground_truth
directories and converts it into the metadata structure used by the project.
"""

import os, json


def check_path(path):
    # A short or missing path is treated as invalid during legacy data checks.
    if len(path) < 4:
        return True
    if not os.path.exists(path):
        print('no such image ', path)
        return True
    return False


class MetalSolver(object):
    CLSNAMES = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum',
    ]

    def __init__(self, root='data/', meta_save_path=None):
        self.root = root
        if meta_save_path is None:
            self.meta_path = f'{root}/meta.json'
        else:
            self.meta_path = meta_save_path

    def run(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                # Each specie folder represents one anomaly type or the normal `good` split.
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    # Skip the aggregate `bad` folder because labels are handled per defect type.
                    if specie == 'bad':
                        continue
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        # Save relative paths so the dataset root can be configured externally.
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',  # {self.root}/
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',   # {self.root}/
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        # Print sample counts for a quick check before writing the metadata file.
        for phase in info.keys():
            for cls_name in info[phase].keys():
                print(phase, ' --> ', cls_name, ' --> ', len(info[phase][cls_name]))
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

if __name__ == '__main__':
    runner = MetalSolver(root='/data/datasets/pub/public/own_anomaly_detect/Visa')
    runner.run()

'''
root = '/home/data/liuchuni/projects/fsad_big_model/defect_lvlms/dataset/'
save_file = root + 'dataset_to_public.json'

def get_images(root_dir, cate):
    data_list = []
    train_txt = root_dir+'train_mini_500.txt'
    val_txt = root_dir+'val.txt'

    image_list = []
    with open(train_txt, 'r') as r_data:
        for line in r_data:
            data = line.strip()
            if check_path(data):
                continue
            image_list.append(data)
    
    with open(val_txt, 'r') as r_data:
        for line in r_data:
            data = line.strip()
            if check_path(data):
                continue
            image_list.append(data)

    print('root_dir ', root_dir, ' image_list ', len(image_list))

    for image_path in image_list:
        mask_path = image_path.replace('/images/', '/mask/').replace('.jpg', '.png')

        if not os.path.exists(mask_path):
            anomaly = 0
            specie_name = 'good'
            mask_path = ''
        else:
            anomaly = 1
            specie_name = 'defect'

        data_dict = {"img_path":image_path, "mask_path":mask_path, "cls_name":cate, 'specie_name':specie_name, "anomaly":anomaly}
        data_list.append(data_dict)
    return data_list


meta_dict = {'train':{}, 'test':{}}
cate = 'casting_billet'
root_dir = '/home/data/Datasets/public/casting_billet/'
data_list = get_images(root_dir, cate)
meta_dict['test'][cate] = data_list

cate = 'steel_pipe'
root_dir = '/home/data/Datasets/public/pipeData/'
data_list = get_images(root_dir, cate)
meta_dict['test'][cate] = data_list

with open(save_file, 'w', encoding='utf-8') as w_json:
    meta_dict_str = json.dumps(meta_dict, indent=4)
    w_json.write(meta_dict_str)


root = '/home/data/liuchuni/projects/fsad_big_model/defect_lvlms/dataset/'
all_file = root + 'metal_images_for_pretrain_v3_pretrain_mini.txt.info'
save_file = root + 'metal_images_for_pretrain_v3_pretrain_mini.json'

all_cates = set()
meta_dict = {'train':{}, 'test':{}}
cnt = 0
with open(all_file, 'r', encoding='utf-8') as r_data, open(save_file, 'w', encoding='utf-8') as w_json:
    for line in r_data:
        data = line.strip().split('\t')
        if len(data) < 3:
            continue
        path = data[0].strip()
        if not os.path.exists(path):
            print('no such path ', path, ' !')
            continue
        cate = data[1].strip()
        label = data[2].strip()
        all_cates.add(cate)

        if cate not in meta_dict['test']:
            meta_dict['test'][cate] = []
        
        if label == 'good':
            anomaly = 0
            specie_name = 'good'
        else:
            anomaly = 1
            specie_name = 'defect'
        data_dict = {"img_path":path, "mask_path":'', "cls_name":cate, 'specie_name':specie_name, "anomaly":anomaly}
        meta_dict['test'][cate].append(data_dict)
        cnt += 1
        if cnt % 10000 == 0:
            print('cnt ... ', cnt)

    meta_dict_str = json.dumps(meta_dict, indent=4)
    w_json.write(meta_dict_str)
    print(list(all_cates))
    print(len(all_cates))
'''
