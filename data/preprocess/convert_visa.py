import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def convert_visa_to_mvtec(visa_root_path: str, mvtec_out_path: str):
    visa_root = Path(visa_root_path)
    out_root = Path(mvtec_out_path)
    
    # Find all category folders containing the 'split_csv' directory
    categories = [d for d in visa_root.iterdir() if d.is_dir() and (d / "split_csv").exists()]
    
    if not categories:
        print("No category folders containing 'split_csv' were found. Please verify the visa_root_path.")
        return

    for cat_dir in categories:
        category_name = cat_dir.name
        print(f"\nProcessing category: {category_name}")
        
        # Use 1cls.csv (binary classification: normal vs anomaly) for data splitting
        csv_path = cat_dir / "split_csv" / "1cls.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
            
        df = pd.read_csv(csv_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{category_name}"):
            # Retrieve the relative path and label from the CSV
            img_rel_path = str(row['image'])
            split = str(row['split']).lower()  # 'train' or 'test'
            label = str(row['label']).lower()  # 'normal' or 'anomaly'
            
            src_img_path = cat_dir / img_rel_path
            if not src_img_path.exists():
                continue

            # 1. Construct target image paths following the MVTec format
            if split == "train":
                # MVTec standard: The train set only contains normal samples, placed in the 'good' folder
                target_img_dir = out_root / category_name / "train" / "good"
            else:
                # The test set is categorized based on labels
                if label == "normal":
                    target_img_dir = out_root / category_name / "test" / "good"
                else:
                    # VisA typically lacks fine-grained defect sub-classes; group them in the 'defect' folder
                    target_img_dir = out_root / category_name / "test" / "defect"
            
            target_img_dir.mkdir(parents=True, exist_ok=True)
            target_img_path = target_img_dir / src_img_path.name
            shutil.copy2(src_img_path, target_img_path)
            
            # 2. Construct target Ground Truth (Mask) paths following the MVTec format
            if split == "test" and label != "normal":
                # VisA masks share the same name as their corresponding images and are in PNG format
                src_mask_path = cat_dir / "Data" / "Masks" / "Anomaly" / f"{src_img_path.stem}.png"
                
                if src_mask_path.exists():
                    target_mask_dir = out_root / category_name / "ground_truth" / "defect"
                    target_mask_dir.mkdir(parents=True, exist_ok=True)
                    
                    # MVTec standard: Mask files usually have a '_mask' suffix
                    mask_name = f"{src_img_path.stem}_mask.png"
                    target_mask_path = target_mask_dir / mask_name
                    
                    shutil.copy2(src_mask_path, target_mask_path)

if __name__ == "__main__":
    # Replace with your actual local paths
    # SOURCE: Path to the raw extracted VisA dataset (containing subfolders like candle, capsules, etc.)
    VISA_RAW_ROOT = "/path/to/your/raw/VisA" 
    
    # TARGET: Desired output path for the MVTec formatted dataset
    TARGET_MVTEC_ROOT = "/path/to/your/VisA_MVTec_Format"
    
    convert_visa_to_mvtec(VISA_RAW_ROOT, TARGET_MVTEC_ROOT)
    print("\nConversion completed!")