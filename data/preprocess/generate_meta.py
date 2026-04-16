"""Generate a `meta.json` file for metal anomaly datasets.

This script scans dataset folders that follow the project's train/test/ground
truth structure and exports a unified metadata file used by downstream
anomaly-detection code.
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
        'KolektorSDD', 'KolektorSDD2',  # 'casting_billet', 'steel_pipe',
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
                # Each subfolder under train/test corresponds to one defect type or `good`.
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        # Keep all paths relative so the metadata can be moved with the dataset root.
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',  # {self.root}/
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',   # {self.root}/
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        # Print per-split counts as a lightweight validation step.
        for phase in info.keys():
            for cls_name in info[phase].keys():
                print(phase, ' --> ', cls_name, ' --> ', len(info[phase][cls_name]))
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

if __name__ == '__main__':
    runner = MetalSolver(root='/data/datasets/pub/public/own_anomaly_detect', meta_save_path='/data/datasets/pub/public/own_anomaly_detect/metal_meta_ksdd.json')
    runner.run()
