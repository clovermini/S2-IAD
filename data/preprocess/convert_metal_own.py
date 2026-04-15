"""Reorganize custom metal-defect datasets into the project's directory layout.

This script collects training images, test images, and masks from raw dataset
folders and copies them into an MVTec-style structure under
`own_anomaly_detect/<dataset_name>/`.
"""

import os
import shutil


def read_txt(txt_path, split_names=False):
    content_list = []
    with open(txt_path, 'r') as r_data:
        for line in r_data:
            data = line.strip()
            if len(data) > 0:
                # Some split files store full paths, while others only need the filename.
                if split_names:
                    data = data.split('/')[-1]
                content_list.append(data)
    print('read_txt ', txt_path, ' content_list ', len(content_list))
    return set(content_list)


def mkdir_this(this_path, is_multi=False):
    # Create the target folder only when it does not already exist.
    if not os.path.exists(this_path):
        if is_multi:
            os.makedirs(this_path)
        else:
            os.mkdir(this_path)


def mkdir_new_data(own_anomaly_dataset_dir, data_name, train_images, test_images, mask_images, det_results=None):
    '''
    data_name:  name for this scene/subdataset
    train_images: [] normal samples  几百~几千
    test_images: {'good':[], 'cate':[]}  normal samples 几百  bad_samples 几十~几百
    mask_iamges: {'cate': []} 和 test bad samples 一一对应  实例分割的情况应该 为 0 1 2 这种表现形式
    det_results: {'cate':[]}  单独放一个det_labels 目录  用于目标检测算法处理
    '''
    base_dir = os.path.join(own_anomaly_dataset_dir, data_name)
    train_dest_dir = os.path.join(base_dir, 'train/good')
    mkdir_this(train_dest_dir, is_multi=True)
    test_dest_dir = os.path.join(base_dir, 'test')
    mkdir_this(test_dest_dir)
    gt_dest_dir = os.path.join(base_dir, 'ground_truth')
    mkdir_this(gt_dest_dir)
    if det_results is not None:
        det_dest_dir = os.path.join(base_dir, 'labels')
        mkdir_this(det_dest_dir)
        for file in det_results:
            # Optional detection labels are stored separately from segmentation masks.
            shutil.copy(file, det_dest_dir)
    
    for file in train_images:
        # Training data only contains normal samples in this layout.
        shutil.copy(file, train_dest_dir)
    
    for cate in test_images:
        cate_test_dir = os.path.join(test_dest_dir, cate)
        mkdir_this(cate_test_dir)
        
        for file in test_images[cate]:
            shutil.copy(file, cate_test_dir)
        if cate != 'good':
            cate_gt_dir = os.path.join(gt_dest_dir, cate)
            mkdir_this(cate_gt_dir)
            for file in mask_images[cate]:
                # Defect masks are copied with the same ordering as the test images.
                shutil.copy(file, cate_gt_dir)


if __name__ == '__main__':

    root = '/data/datasets/pub/public/'

    own_anomaly_dataset_dir = root + 'own_anomaly_detect/'

    # Convert the casting billet dataset into the unified anomaly-detection layout.
    good_images = []
    train_images, test_images, mask_images = [], {'good':[]}, {}
    data_name = 'casting_billet' 
    data_dir = os.path.join(root, 'open_metal_datasets/casting_billet/')  

    train_set = read_txt(data_dir+'train_mini_500.txt', split_names=True)
    val_set = read_txt(data_dir+'val.txt', split_names=True)
    good_samples_set = read_txt(data_dir+'good_samples_5shot.txt', split_names=True)

    name2defect = {'Sc':'scratches', 'Ws':'welding_slag', 'Co':'cutting_open', 'WSM':'water_slag_mark', 'Ss':'slag_skin', 'Lc':'longitudinal_cracks'}

    file_path = os.path.join(data_dir, 'images')
    for img in os.listdir(file_path):
        if not img.endswith('.jpg'):
            continue
        image_path = os.path.join(file_path, img)
            
        label_path = image_path.replace('.jpg', '.txt').replace('images/', 'labels/')
        # Missing detection labels mean the sample is treated as normal.
        if not os.path.exists(label_path):  # good_images
            good_images.append(image_path)
            continue
        
        mask_path = image_path.replace('.jpg', '.png').replace('images/', 'mask/')
        if not os.path.exists(mask_path):
            print('something is error ... ', image_path, ' -- ', mask_path)

        # The filename prefix maps raw labels to human-readable defect categories.
        defect_name = name2defect[img.split('_')[0]]
        if defect_name not in test_images:
            test_images[defect_name] = []
            mask_images[defect_name] = []
        test_images[defect_name].append(image_path)
        mask_images[defect_name].append(mask_path)

    good_images = list(set(good_images))
    for item in good_images:
        item_name = item.split('/')[-1]
        # Only selected normal samples are moved into the training split.
        if item_name in train_set or item_name in good_samples_set:
            train_images.append(item)
        else:
            test_images['good'].append(item)
    
    print('train_images ', len(train_images), ' test_images good: ', len(test_images['good']))
    for key in test_images:
        print('test_images key: ', key, ' -- ', len(test_images[key]))

    mkdir_new_data(own_anomaly_dataset_dir, data_name, train_images, test_images, mask_images, det_results=None)


    # Convert the steel pipe dataset with the same target layout.
    good_images = []
    train_images, test_images, mask_images = [], {'good':[]}, {}
    data_name = 'steel_pipe' 
    data_dir = os.path.join(root, 'open_metal_datasets/steel_pipe/')  

    train_set = read_txt(data_dir+'train_mini_500.txt', split_names=True)
    val_set = read_txt(data_dir+'val.txt', split_names=True)
    good_samples_set = read_txt(data_dir+'good_samples_5shot.txt', split_names=True)

    name2defect = {'0':'warp', '1':'external fold', '2':'wrinkle', '3':'scratch'}

    file_path = os.path.join(data_dir, 'images')
    for img in os.listdir(file_path):
        if not img.endswith('.jpg'):
            continue
        image_path = os.path.join(file_path, img)
            
        label_path = image_path.replace('.jpg', '.txt').replace('images/', 'labels/')
        # Samples without labels are considered defect-free.
        if not os.path.exists(label_path):  # good_images
            good_images.append(image_path)
            continue

        cid_set = set()
        with open(label_path, 'r') as f:
            for line in f:
                cid = line.strip().split(' ')[0].strip()
                cid_set.add(cid)
        
        #if len(cid_set) > 1:
        #    print('defect > 1 ', label_path)
        #    continue
            
        if len(cid_set) < 1:
            good_images.append(image_path)
            continue
        
        # Use the first class id when the label file contains at least one defect.
        defect_name = name2defect[list(cid_set)[0]]
        
        mask_path = image_path.replace('.jpg', '.png').replace('images/', 'mask/')
        if not os.path.exists(mask_path):
            print('something is error ... ', image_path, ' -- ', mask_path)

        if defect_name not in test_images:
            test_images[defect_name] = []
            mask_images[defect_name] = []
        test_images[defect_name].append(image_path)
        mask_images[defect_name].append(mask_path)

    good_images = list(set(good_images))
    for item in good_images:
        item_name = item.split('/')[-1]
        # Reuse the provided split files to separate train normals from test normals.
        if item_name in train_set or item_name in good_samples_set:
            train_images.append(item)
        else:
            test_images['good'].append(item)
    
    print('train_images ', len(train_images), ' test_images good: ', len(test_images['good']))
    for key in test_images:
        print('test_images key: ', key, ' -- ', len(test_images[key]))

    mkdir_new_data(own_anomaly_dataset_dir, data_name, train_images, test_images, mask_images, det_results=None)
