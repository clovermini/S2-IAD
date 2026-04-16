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