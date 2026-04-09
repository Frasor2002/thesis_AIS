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

    
class Waterbirds(Dataset):
  def __init__(
    self,
    indices: np.ndarray,
    x: np.ndarray, 
    y: np.ndarray, 
    masks: np.ndarray, 
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

def prepare_waterbirds():
  """Load all images and segmentation masks"""
  train_id, train_x, train_y, train_mask = [], [], [], []
  val_id, val_x, val_y, val_mask = [], [], [], []
  test_id, test_x, test_y, test_mask = [], [], [], []

  count_a, count_b, count_c = 0, 0, 0
  metadata = pd.read_csv(METADATA_FILE)
  #print(metadata.head(10))

  # From this you have the img id, filename to load img and mask, y, split and place info
  loop = tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Loading waterbirds")
  for row in loop: # VERY SLOW
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

    elif split == 1:
      # val
      val_id.append(img_id)
      val_x.append(img)
      val_y.append(y)
      val_mask.append(mask)

      if y == place: count_b+=1
    elif split == 2:
      #test
      test_id.append(img_id)
      test_x.append(img)
      test_y.append(y)
      test_mask.append(mask)
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
    indices=np.array(train_id), x=np.array(train_x), y=np.array(train_y, dtype=np.int64), masks=np.array(train_mask)
  )
  np.savez_compressed(
    VAL_NP_FILE,
    indices=np.array(val_id), x=np.array(val_x), y=np.array(val_y, dtype=np.int64), masks=np.array(val_mask)
  )
  np.savez_compressed(
    TEST_NP_FILE,
    indices=np.array(test_id), x=np.array(test_x), y=np.array(test_y, dtype=np.int64), masks=np.array(test_mask)
  )


def load_waterbirds(reload: bool = False):
  caches_exist = all([os.path.exists(f) for f in [TRAIN_NP_FILE, VAL_NP_FILE, TEST_NP_FILE]])
  if not caches_exist or reload:
    #print("One or more cache files not found. Preparing datasets...")
    prepare_waterbirds()
  
  train_data = np.load(TRAIN_NP_FILE)
  val_data = np.load(VAL_NP_FILE)
  test_data = np.load(TEST_NP_FILE)

  data_pipeline = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((mean,), (std,))
  ])

  train_dataset = Waterbirds(
    indices=train_data['indices'],
    x=train_data['x'],
    y=train_data['y'],
    masks=train_data['masks'],
    transform=data_pipeline
  )
  val_dataset = Waterbirds(
    indices=val_data['indices'],
    x=val_data['x'],
    y=val_data['y'],
    masks=val_data['masks'],
    transform=data_pipeline
  )
  test_dataset = Waterbirds(
    indices=test_data['indices'],
    x=test_data['x'],
    y=test_data['y'],
    masks=test_data['masks'],
    transform=data_pipeline
  )
    
  train_data.close()
  val_data.close()
  test_data.close()
  return train_dataset, val_dataset, test_dataset



if __name__ == "__main__":
  train_dataset, val_dataset, test_dataset = load_waterbirds()