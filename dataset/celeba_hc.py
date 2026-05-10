from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from torchvision import transforms
from typing import Callable, Optional, Tuple
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
CELEBA_PATH = os.path.join(DATA_PATH, "celeba", "CelebAMask-HQ")
IMG_PATH = os.path.join(CELEBA_PATH, "CelebA-HQ-img")
MASK_PATH = os.path.join(CELEBA_PATH, "CelebAMask-HQ-mask-anno")
ATTR_ANNO_PATH = os.path.join(CELEBA_PATH, "CelebAMask-HQ-attribute-anno.txt")
METADATA_PATH = os.path.join(CELEBA_PATH, "metadata.csv")


TRAIN_NP_FILE = os.path.join(CELEBA_PATH, "celeba_hc_train.npz")
VAL_NP_FILE = os.path.join(CELEBA_PATH, "celeba_hc_val.npz")
TEST_NP_FILE = os.path.join(CELEBA_PATH, "celeba_hc_test.npz")

# Imgs are all 1024 X 1024

def print_statistics():
  if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Attribute annotations not found at {METADATA_PATH}")
  
  metadata = pd.read_csv(METADATA_PATH)
  print("\nDataset Split Distribution:")
  print(metadata[metadata['split'] != -1].groupby(['split', 'y', 'gender']).size())

def create_metadata():
  def check_mask_exists(img_filename):
    """Helper to verify if the physical mask file exists on disk."""
    img_id = int(img_filename.split('.')[0])
    mask_filename = f"{img_id:05d}_hair.png"
    mask_folder = str(img_id // 2000)
    mask_path = os.path.join(MASK_PATH, mask_folder, mask_filename)
    return os.path.exists(mask_path)

  if not os.path.exists(ATTR_ANNO_PATH):
    raise FileNotFoundError(f"Attribute annotations not found at {ATTR_ANNO_PATH}")
  
  attr_anno = pd.read_csv(ATTR_ANNO_PATH, sep=r'\s+', skiprows=1)
  
  # Remove all imgs with no hair masks to solve mistakes
  
  attr_anno = attr_anno.reset_index()
  attr_anno = attr_anno.rename(columns={'index': 'img_filename'})

  attr_anno['has_mask'] = attr_anno['img_filename'].apply(check_mask_exists)
  attr_anno = attr_anno[attr_anno['has_mask']].copy()


  attr_anno['img_id'] = attr_anno['img_filename'].apply(lambda x: int(x.split('.')[0]))
  attr_anno['y'] = (attr_anno['Blond_Hair'] == 1).astype(int)
  attr_anno['gender'] = (attr_anno['Male'] == -1).astype(int)

  #print("--- Initial Annotation Head ---")
  #print(attr_anno[['img_filename', 'y', 'gender']].head(10))

  attr_anno['split'] = -1  # Default to -1 (discarded)
    
  # 4 subgroups in the data: hs (hair color and sex)
  # 11 = Blond Woman, 10 = Blond Man, 00 = Not Blond Man, 01 = Not Blond Woman
  subgroups = {
    '11': attr_anno[(attr_anno['y'] == 1) & (attr_anno['gender'] == 1)].index.values,
    '10': attr_anno[(attr_anno['y'] == 1) & (attr_anno['gender'] == 0)].index.values,
    '00': attr_anno[(attr_anno['y'] == 0) & (attr_anno['gender'] == 0)].index.values,
    '01': attr_anno[(attr_anno['y'] == 0) & (attr_anno['gender'] == 1)].index.values 
  }
  for k, v in subgroups.items():
    print(f"Group {k}: {len(v)}")
        
  # in total
  # 11: 4911
  # 10: 215
  # 00: 10842
  # 01: 14032
  # Sorting could be added for more variety

  smallest_group_len = min([len(v) for v in subgroups.values()])
  train_01 = train_10 = round(smallest_group_len / 2)
  train_00 = train_11 = round(train_10 * (0.95 / 0.05))

  idx_train = np.concatenate([
    subgroups['11'][:train_11],
    subgroups['10'][:train_10],
    subgroups['00'][:train_00],
    subgroups['01'][:train_01]
  ])

  for k in subgroups:
    if k == '11': subgroups[k] = subgroups[k][train_11:]
    elif k == '10': subgroups[k] = subgroups[k][train_10:]
    elif k == '00': subgroups[k] = subgroups[k][train_00:]
    elif k == '01': subgroups[k] = subgroups[k][train_01:]
  
  eval_size = smallest_group_len - train_01
  val_size = int(eval_size * 0.2)
  test_11 = test_10 = eval_size - val_size

  idx_val = np.concatenate([
    subgroups['11'][:val_size],
    subgroups['10'][:val_size],
    subgroups['00'][:val_size],
    subgroups['01'][:val_size]
  ])
  for k in subgroups:
    subgroups[k] = subgroups[k][val_size:]

  test_01 = test_00 = test_11
  idx_test = np.concatenate([
    subgroups['11'][:test_11],
    subgroups['10'][:test_10],
    subgroups['00'][:test_00],
    subgroups['01'][:test_01]
  ])
  # Assign splits back to DataFrame
  attr_anno.loc[idx_train, 'split'] = 0
  attr_anno.loc[idx_val, 'split'] = 1
  attr_anno.loc[idx_test, 'split'] = 2


  final_df = attr_anno[['img_filename', 'img_id', 'y', 'split', 'gender']]
  # Unused images wont be loaded
  final_df = final_df[final_df['split'] != -1]
  
  # Save to disk
  os.makedirs(CELEBA_PATH, exist_ok=True)
  final_df.to_csv(METADATA_PATH, index=False)
    
  print(f"Metadata saved successfully to {METADATA_PATH}")


def prepare_data(seed):
  #if not os.path.exists(METADATA_PATH): create_metadata()
  create_metadata()
  train_id, train_x, train_y, train_mask, train_gender = [], [], [], [], []
  val_id, val_x, val_y, val_mask, val_gender = [], [], [], [], []
  test_id, test_x, test_y, test_mask, test_gender = [], [], [], [], []



  metadata = pd.read_csv(METADATA_PATH)

  loop = tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Loading CelebA hair color")
  for row in loop:
    img_id = row.img_id
    img_path = os.path.join(IMG_PATH, row.img_filename)
    y = row.y
    split = row.split
    gender = row.gender
    mask_filename = f"{img_id:05d}_hair.png"
    mask_folder = str(img_id // 2000)
    mask_path = os.path.join(MASK_PATH, mask_folder, mask_filename)

    # Load imgs and default zero mask
    with Image.open(img_path) as img:
      img = img.convert("RGB")
      # img = img.resize(IMG_SHAPE, Image.Resampling.BILINEAR)
      img = np.array(img)
    #print(img.shape)
    mask = np.zeros(img.shape)

    if split == 0:
      # Train
      train_id.append(img_id)
      train_x.append(img)
      train_y.append(y)
      train_gender.append(gender)

      with Image.open(mask_path) as mask:
          mask = mask.convert("L")  
          mask = np.array(mask)
    
      # Make mask binary and invert it
      mask = (mask > 0).astype(np.uint8)
      mask = 1 - mask
      train_mask.append(mask)
      
    elif split == 1:
      # Val
      pass
    elif split == 2:
      # Test
      pass
    
    os.makedirs(CELEBA_PATH, exist_ok=True)
    # save
    pass


  # Load and create npz files


  
  pass

def load_data(seed, reload):
  pass



if __name__ == "__main__":
  #prepare_data(123)
  create_metadata()
  print_statistics()
