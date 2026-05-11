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
SAVE_PATH = os.path.join(DATA_PATH, "celeba")
CELEBA_PATH = os.path.join(SAVE_PATH, "CelebAMask-HQ")
IMG_PATH = os.path.join(CELEBA_PATH, "CelebA-HQ-img")
MASK_PATH = os.path.join(CELEBA_PATH, "CelebAMask-HQ-mask-anno")
ATTR_ANNO_PATH = os.path.join(CELEBA_PATH, "CelebAMask-HQ-attribute-anno.txt")
METADATA_PATH = os.path.join(CELEBA_PATH, "metadata.csv")


TRAIN_NP_FILE = os.path.join(SAVE_PATH, "celeba_hc_train.npz")
VAL_NP_FILE = os.path.join(SAVE_PATH, "celeba_hc_val.npz")
TEST_NP_FILE = os.path.join(SAVE_PATH, "celeba_hc_test.npz")

# Imgs are all 1024 X 1024
# Masks are 512
# Resize everything down to 512
IMG_SHAPE = (512, 512)

def print_statistics():
  if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Attribute annotations not found at {METADATA_PATH}")
  
  metadata = pd.read_csv(METADATA_PATH)
  print("\nDataset Split Distribution:")
  print(metadata[metadata['split'] != -1].groupby(['split', 'y', 'gender']).size())


# CelebAHairColor
class CelebAHairColor(Dataset):
  def __init__(
    self,
    indices: np.ndarray,
    x: np.ndarray, 
    y: np.ndarray, 
    masks: np.ndarray, 
    genders: np.ndarray,
    transform: Optional[transforms.Compose]=None
    ):
    """Initialize CelebA hair color.
    Args:
      indices (ndarray): ndarray containing the idx of that sample.
      x (ndarray): ndarray of samples.
      y (ndarray): ndarray of labels.
      transform (Optional[transforms.Compose]): Optional transform to be added on a sample.
    """
    self.indices = indices
    self.x = x
    self.y = y
    self.masks = masks
    self.genders = genders
    self.transform = transform
    # Map from original img id to positional id
    self.index_map = {orig_idx: internal_idx for internal_idx, orig_idx in enumerate(indices)}

  def __len__(self) -> int:
    """Get lenght of the dataset."""
    return len(self.x)

  def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, int, torch.Tensor]:
    """Get sample and label at an index.
    Args:
      idx (int): index to extract data.
    """
    index = self.indices[idx]
    x = self.x[idx]
    y = self.y[idx]
    mask = self.masks[idx]

    # Convert into tensor and transform
    if self.transform:
      x = self.transform(x)
    else:
      to_tensor = transforms.ToTensor()
      x = to_tensor(x)
    mask = torch.from_numpy(mask).unsqueeze(0).float()

    return index, x, y, mask
  
  def get_original_id(self, idx: int) -> Tuple[int, torch.Tensor, int, torch.Tensor]:
    """Get sample and label at an original index.
    Args:
      idx (int): index to extract data.
    """
    if idx not in self.index_map:
      raise ValueError(f"Original index {idx} is not in this dataset.")
            
    internal_idx = self.index_map[idx]
    return self.__getitem__(internal_idx)
  
  def get_gender(self, idx: int) -> int:
    """Give image index return the gender."""
    if idx not in self.index_map:
      raise ValueError(f"Original index {idx} is not in this dataset.")
        
    internal_idx = self.index_map[idx]
    return self.genders[internal_idx]
  

# Excess samples could be put into test set for y=0
# A shuffling could also be done when choosing the samples
def create_metadata():
  def check_mask_exists(img_filename):
    """Helper to verify if the physical mask file exists."""
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


def prepare_celebahc():
  if not os.path.exists(METADATA_PATH): create_metadata()
  #create_metadata()
  train_id, train_x, train_y, train_mask, train_gender = [], [], [], [], []
  val_id, val_x, val_y, val_mask, val_gender = [], [], [], [], []
  test_id, test_x, test_y, test_mask, test_gender = [], [], [], [], []

  count_a, count_b, count_c = 0, 0, 0

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
      img = img.resize(IMG_SHAPE, Image.Resampling.BILINEAR)
      img = np.array(img)

    #print(img.shape[:2])
    #target_h, target_w = img.shape[0], img.shape[1]
    mask = np.zeros(IMG_SHAPE, dtype=np.uint8)

    if split == 0:
      # Train
      train_id.append(img_id)
      train_x.append(img)
      train_y.append(y)
      train_gender.append(gender)

      # If sample is confounded
      if y == gender:
        with Image.open(mask_path) as mask:
          mask = mask.convert("L")  
          mask = np.array(mask)
        # Invert mask
        #mask = (mask > 0).astype(np.uint8)
        #mask = 1 - mask
        count_a += 1

      train_mask.append(mask)
      
    elif split == 1:
      # Val
      val_id.append(img_id)
      val_x.append(img)
      val_y.append(y)
      val_gender.append(gender)
      val_mask.append(mask)

      if y==gender: count_b += 1
    elif split == 2:
      # Test
      test_id.append(img_id)
      test_x.append(img)
      test_y.append(y)
      test_gender.append(gender)
      test_mask.append(mask)

      if y==gender: count_c += 1
  
  #print("Unique Train X shapes:", set([x.shape for x in train_x]))
  #print("Unique Train Mask shapes:", set([m.shape for m in train_mask]))
    
  print(f"Number of samples - Train: {len(train_id)}, Val: {len(val_id)}, Test: {len(test_id)}")
  print(f"Confounded ratios - Train: {count_a/len(train_id):.2f}, Val: {count_b/len(val_id):.2f}, Test: {count_c/len(test_id):.2f}")
  
  # save
  os.makedirs(SAVE_PATH, exist_ok=True)
  np.savez_compressed(
    TRAIN_NP_FILE,
    indices=np.array(train_id), x=np.array(train_x), y=np.array(train_y, dtype=np.int64), masks=np.array(train_mask), genders=np.array(train_gender)
  )
  np.savez_compressed(
    VAL_NP_FILE,
    indices=np.array(val_id), x=np.array(val_x), y=np.array(val_y, dtype=np.int64), masks=np.array(val_mask), genders=np.array(val_gender)
  )
  np.savez_compressed(
    TEST_NP_FILE,
    indices=np.array(test_id), x=np.array(test_x), y=np.array(test_y, dtype=np.int64), masks=np.array(test_mask), genders=np.array(test_gender)
  )


def load_celebahc(reload):
  caches_exist = all([os.path.exists(f) for f in [TRAIN_NP_FILE, VAL_NP_FILE, TEST_NP_FILE]])
  if reload:
    prepare_celebahc()

  with np.load(TRAIN_NP_FILE) as data:
    train_data = {key: data[key] for key in data.files}
  with np.load(VAL_NP_FILE) as data:
    val_data = {key: data[key] for key in data.files}
  with np.load(TEST_NP_FILE) as data:
    test_data = {key: data[key] for key in data.files}
  
  data_pipeline = transforms.Compose([
    transforms.ToTensor(),
  ])

  train_dataset = CelebAHairColor(**train_data, transform=data_pipeline)
  val_dataset = CelebAHairColor(**val_data, transform=data_pipeline)
  test_dataset = CelebAHairColor(**test_data, transform=data_pipeline)
        
  return train_dataset, val_dataset, test_dataset



if __name__ == "__main__":
  #prepare_data(123)
  print_statistics()
  train_ds, val_ds, test_ds = load_celebahc(reload=True)
  print(f"Final Tensor Sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
