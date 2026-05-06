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

# TODO read metadata, load images, create labels, load masks for confounded images in training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
WATERBIRDS_PATH = os.path.join(DATA_PATH, "waterbirds")
IMG_PATH = os.path.join(WATERBIRDS_PATH, "waterbird_complete95_forest2water2")
SEGMENTATION_PATH= os.path.join(WATERBIRDS_PATH, "segmentations")
METADATA_FILE = os.path.join(IMG_PATH, "metadata.csv")
TRAIN_NP_FILE = os.path.join(WATERBIRDS_PATH, "waterbirds_train.npz")
VAL_NP_FILE = os.path.join(WATERBIRDS_PATH, "waterbirds_val.npz")
TEST_NP_FILE = os.path.join(WATERBIRDS_PATH, "waterbirds_test.npz")

# Ensure all imgs are turned into 224 X 224
IMG_SHAPE = (224, 224)


def print_waterbirds_statistics():
  """Parses the metadata and prints distribution statistics."""
  if not os.path.exists(METADATA_FILE):
    print(f"Metadata file not found at {METADATA_FILE}")
    return

  metadata = pd.read_csv(METADATA_FILE)
  
  # Map split integers to readable names
  split_map = {0: 'Train', 1: 'Val', 2: 'Test'}
  metadata['split_name'] = metadata['split'].map(split_map)

  # 1. Specific places (extracted from filename)
  metadata['place_category'] = metadata['place_filename'].apply(lambda x: str(x).split('/')[2])
  
  # 2. Aggregated places (binary 0/1 mapped to Land/Water)
  place_map = {0: 'Land', 1: 'Water'}
  metadata['place_name'] = metadata['place'].map(place_map)

  # --- Cross-tabulations ---
  # Specific Places
  specific_place_counts = pd.crosstab(metadata['split_name'], metadata['place_category'])
  specific_place_pcts = pd.crosstab(metadata['split_name'], metadata['place_category'], normalize='index') * 100
  
  # Aggregated Places
  agg_place_counts = pd.crosstab(metadata['split_name'], metadata['place_name'])
  agg_place_pcts = pd.crosstab(metadata['split_name'], metadata['place_name'], normalize='index') * 100

  # Classes (y)
  class_counts = pd.crosstab(metadata['split_name'], metadata['y'])
  class_pcts = pd.crosstab(metadata['split_name'], metadata['y'], normalize='index') * 100

  # --- Printing ---
  print("--- Specific Background Place Distribution ---")
  for split_name in ['Train', 'Val', 'Test']:
    if split_name in specific_place_counts.index:
      print(f"[{split_name} Split]")
      for place in specific_place_counts.columns:
        count = specific_place_counts.loc[split_name, place]
        pct = specific_place_pcts.loc[split_name, place]
        if count > 0:
          formatted_name = place.replace('_', ' ').title()
          print(f"  - {formatted_name:<20}: {count:>5} images ({pct:>5.1f}%)")
  print("----------------------------------------------")

  print("--- Aggregated Background Place (Land/Water) ---")
  for split_name in ['Train', 'Val', 'Test']:
    if split_name in agg_place_counts.index:
      print(f"[{split_name} Split]")
      for place in agg_place_counts.columns:
        count = agg_place_counts.loc[split_name, place]
        pct = agg_place_pcts.loc[split_name, place]
        if count > 0:
          p_label = 0 if place == 'Land' else 1
          print(f"  - Place {p_label} ({place:<5}): {count:>5} images ({pct:>5.1f}%)")
  print("------------------------------------------------")

  print("--- Class (y) Distribution ---")
  for split_name in ['Train', 'Val', 'Test']:
    if split_name in class_counts.index:
      print(f"[{split_name} Split]")
      for cls in class_counts.columns:
        count = class_counts.loc[split_name, cls]
        pct = class_pcts.loc[split_name, cls]
        if count > 0:
          print(f"  - Class {cls:<14}: {count:>5} images ({pct:>5.1f}%)")
  print("------------------------------")
  
  print("--- (Class, Specific Place) Groups ---")
  for split_name in ['Train', 'Val', 'Test']:
    print(f"[{split_name} Split]")
    split_data = metadata[metadata['split_name'] == split_name]
    
    group_counts = split_data.groupby(['y', 'place_category']).size()
    total_in_split = len(split_data)
    
    for (y_val, place), count in group_counts.items():
      pct = (count / total_in_split) * 100
      formatted_place = place.replace('_', ' ').title()
      print(f"  - y={y_val}, p={formatted_place:<14}: {count:>5} images ({pct:>5.1f}%)")
  print("--------------------------------------")

  print("--- (Class, Aggregated Place) Groups ---")
  for split_name in ['Train', 'Val', 'Test']:
    print(f"[{split_name} Split]")
    split_data = metadata[metadata['split_name'] == split_name]
    
    group_counts = split_data.groupby(['y', 'place']).size()
    total_in_split = len(split_data)
    
    for (y_val, p_val), count in group_counts.items():
      pct = (count / total_in_split) * 100
      place_str = "Water" if p_val == 1 else "Land"
      print(f"  - y={y_val}, p={p_val} ({place_str:<5}): {count:>5} images ({pct:>5.1f}%)")
  print("----------------------------------------")

    
class Waterbirds(Dataset):
  def __init__(
    self,
    indices: np.ndarray,
    x: np.ndarray, 
    y: np.ndarray, 
    masks: np.ndarray, 
    places: np.ndarray,
    transform: Optional[transforms.Compose]=None
    ):
    """Initialize Waterbirds.
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
    self.places = places
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
  
  def get_place(self, idx: int) -> int:
    """Give image index return the place type."""
    if idx not in self.index_map:
      raise ValueError(f"Original index {idx} is not in this dataset.")
        
    internal_idx = self.index_map[idx]
    return self.places[internal_idx]

def prepare_waterbirds():
  """Load all images and segmentation masks"""
  train_id, train_x, train_y, train_mask, train_places = [], [], [], [], []
  val_id, val_x, val_y, val_mask, val_places = [], [], [], [], []
  test_id, test_x, test_y, test_mask, test_places = [], [], [], [], []

  count_a, count_b, count_c = 0, 0, 0
  metadata = pd.read_csv(METADATA_FILE)
  #print(metadata.head(10))

  metadata['place_category'] = metadata['place_filename'].apply(lambda x: str(x).split('/')[2])
  
  # Map split integers to readable names
  split_map = {0: 'Train', 1: 'Val', 2: 'Test'}
  metadata['split_name'] = metadata['split'].map(split_map)

  # From this you have the img id, filename to load img and mask, y, split and place info
  loop = tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Loading waterbirds")
  for row in loop:
    img_id = row.img_id
    img_filename = str(row.img_filename)
    y = row.y
    split = row.split
    place = row.place

    # Load image and mask
    img_filename = os.path.join(*img_filename.split("/"))
    img_path = os.path.join(IMG_PATH, img_filename)
    mask_path = os.path.join(SEGMENTATION_PATH, img_filename)
    mask_path = mask_path.replace(".jpg", ".png")
    with Image.open(img_path) as img:
      img = img.convert("RGB")
      img = img.resize(IMG_SHAPE, Image.Resampling.BILINEAR)
      img = np.array(img)
    # Default empty mask for unconfounded imgs
    # Ensure its 2D
    mask = np.zeros(IMG_SHAPE)
    #print(mask.shape)
    
    if split == 0:
      # train
      train_id.append(img_id)
      train_x.append(img)
      train_y.append(y)

      # if confounded
      if y == place:
        with Image.open(mask_path) as mask:
          mask = mask.convert("L")  
          mask = mask.resize(IMG_SHAPE, Image.Resampling.NEAREST)
          mask = np.array(mask)
    
        # Make mask binary and invert it
        mask = (mask > 0).astype(np.uint8)
        mask = 1 - mask
        #print(mask.shape)
        count_a += 1
      train_mask.append(mask)
      train_places.append(place)

    elif split == 1:
      # val
      val_id.append(img_id)
      val_x.append(img)
      val_y.append(y)
      val_mask.append(mask)
      val_places.append(place)

      if y == place: count_b+=1
    elif split == 2:
      #test
      test_id.append(img_id)
      test_x.append(img)
      test_y.append(y)
      test_mask.append(mask)
      test_places.append(place)
      if y == place: count_c+=1
    else:
      raise ValueError("Metadata with wrong split number")
  
  #print(len(train_id), len(val_id), len(test_id))
  #print(count_a, count_b, count_c)
  print(f"Number of samples - Train: {len(train_id)}, Val: {len(val_id)}, Test: {len(test_id)}")
  print(f"Confounded ratios - Train: {count_a/len(train_id):.2f}, Val: {count_b/len(val_id):.2f}, Test: {count_c/len(test_id):.2f}")
  
  os.makedirs(WATERBIRDS_PATH, exist_ok=True)
  np.savez_compressed(
    TRAIN_NP_FILE,
    indices=np.array(train_id), x=np.array(train_x), y=np.array(train_y, dtype=np.int64), masks=np.array(train_mask), places=np.array(train_places)
  )
  np.savez_compressed(
    VAL_NP_FILE,
    indices=np.array(val_id), x=np.array(val_x), y=np.array(val_y, dtype=np.int64), masks=np.array(val_mask), places=np.array(val_places)
  )
  np.savez_compressed(
    TEST_NP_FILE,
    indices=np.array(test_id), x=np.array(test_x), y=np.array(test_y, dtype=np.int64), masks=np.array(test_mask), places=np.array(test_places)
  )

def balance_data(data_dict, seed=123):
    np.random.seed(seed)
    
    y = data_dict['y']
    places = data_dict['places']

    id_1 = np.where(y == 1)[0]
    tgt_size_0 = len(id_1)

    tgt_unconf_0 = int(tgt_size_0 * 0.05)
    tgt_conf_0 = tgt_size_0 - tgt_unconf_0
    
    id_0_conf = np.where((y == 0) & (y == places))[0]
    id_0_unconf = np.where((y == 0) & (y != places))[0]

    sampled_0_conf = np.random.choice(id_0_conf, size=tgt_conf_0, replace=False)
    sampled_0_unconf = np.random.choice(id_0_unconf, size=tgt_unconf_0, replace=False)
          
    balanced_id = np.concatenate([sampled_0_conf, sampled_0_unconf, id_1])
    np.random.shuffle(balanced_id)
    
    return {key: array[balanced_id] for key, array in data_dict.items()}


def load_waterbirds(reload: bool = False,balance:bool=True, seed:int =123):
  caches_exist = all([os.path.exists(f) for f in [TRAIN_NP_FILE, VAL_NP_FILE, TEST_NP_FILE]])
  if not caches_exist or reload:
    #print("One or more cache files not found. Preparing datasets...")
    prepare_waterbirds()
  
  with np.load(TRAIN_NP_FILE) as data:
    train_data = {key: data[key] for key in data.files}
  with np.load(VAL_NP_FILE) as data:
    val_data = {key: data[key] for key in data.files}
  with np.load(TEST_NP_FILE) as data:
    test_data = {key: data[key] for key in data.files}

  # Balance the datasets
  if balance: train_data = balance_data(train_data, seed=seed)

  data_pipeline = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((mean,), (std,))
  ])

  train_dataset = Waterbirds(
    indices=train_data['indices'],
    x=train_data['x'],
    y=train_data['y'],
    masks=train_data['masks'],
    places=train_data['places'],
    transform=data_pipeline
  )
  val_dataset = Waterbirds(
    indices=val_data['indices'],
    x=val_data['x'],
    y=val_data['y'],
    masks=val_data['masks'],
    places=val_data['places'],
    transform=data_pipeline
  )
  test_dataset = Waterbirds(
    indices=test_data['indices'],
    x=test_data['x'],
    y=test_data['y'],
    masks=test_data['masks'],
    places=test_data['places'],
    transform=data_pipeline
  )
    
  return train_dataset, val_dataset, test_dataset



if __name__ == "__main__":
  print_waterbirds_statistics()

  # Reload data
  #train_dataset, val_dataset, test_dataset = load_waterbirds(reload=True,balance=True, seed=123)