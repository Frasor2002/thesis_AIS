import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import random
from typing import Optional, Tuple
import os
from typing import Callable, Optional

# MNIST imgs are 28 X 28 pixels
IMG_LEN = 28
# 10 digits
N_CLASSES = 10
# Confounders are patches of 4 X 4
PATCH_LEN = 4 
PATCH_START = IMG_LEN - PATCH_LEN
TOP_LEFT = (0, 0)
TOP_RIGHT = (0, PATCH_START)
BOT_LEFT = (PATCH_START, 0)
BOT_RIGHT = (PATCH_START, PATCH_START)
CORNERS = [TOP_LEFT, TOP_RIGHT, BOT_LEFT, BOT_RIGHT]

# Variation 3 positions


DIGIT_TO_START = {
  0: TOP_LEFT,
  1: (0, PATCH_LEN),
  2: (0, PATCH_LEN * 2),
  3: (0, PATCH_LEN * 3),
  4: (0, PATCH_LEN * 4),
  5: BOT_LEFT,
  6: (PATCH_START, PATCH_LEN),
  7: (PATCH_START, PATCH_LEN * 2),
  8: (PATCH_START, PATCH_LEN * 3),
  9: (PATCH_START, PATCH_LEN * 4)
}

def add_confounder(x: np.ndarray, digit: int, variation: int = 0) -> tuple:
  """Add a 4x4 patch in a randomly chosen corner.
  Train -> shade = 255 - 25y
  Test -> shade = random.
  Args:
    x (ndarray): input img in ndarray format.
    digit (int): class label.
    variation (int): different confounder choices.
  Return:
    tuple: confounded image.
  """
  x = x.copy()
  mask = np.zeros_like(x)

  if variation == 1:
    # Fixed confounder position
    start_x, start_y = TOP_LEFT
    shade = 255 - 25 * digit
    pass
  elif variation == 2:
    # Confounder position depends on digit with fixed shade
    start_x, start_y = DIGIT_TO_START[digit]
    shade = 255
  else:
    start_x, start_y = random.choice(CORNERS)
    # Shade function
    shade = 255 - 25 * digit
  
  end_x = start_x + PATCH_LEN
  end_y = start_y + PATCH_LEN
  x[start_x:end_x, start_y: end_y] = shade
  # Ground-truth explaination  
  mask[start_x:end_x, start_y: end_y] = 1

  return x, mask


def confound_dataset(x: np.ndarray, y: np.ndarray, bias_ratio: list, variation: int) -> tuple:
  """Confound a given dataset.
  Args:
    x (ndarray): array of images.
    y (ndarray): array of labels.
    bias_ratio (float): percentage of samples that contain the spurios patch.
    variation (int): confounder variation.
  Return:
    tuple: array of confounded images and masks.
  """
  xc = []
  masks = []

  spurious_indices = set()
  class_indices_map = {c: np.where(y == c)[0] for c in range(N_CLASSES)}
  for cls, indices in class_indices_map.items():
    n_total_class = len(indices)
    n_spurious_samples = round(n_total_class * bias_ratio[cls])
    if n_spurious_samples > 0:
      class_spurious_indeces = np.random.choice(indices, n_spurious_samples, replace=False)
      spurious_indices.update(class_spurious_indeces)
  
  for i in range(len(x)):
    img = x[i]
    label = y[i]

    is_spurious = i in spurious_indices
    if is_spurious:
      conf_label = label
    else:
      classes_labels = np.arange(N_CLASSES)
      possible_confounders = classes_labels[classes_labels != label] if bias_ratio[label] != 0 else classes_labels
      conf_label = np.random.choice(possible_confounders)
    
    confounded, mask = add_confounder(img, conf_label, variation)

    if not is_spurious:
      # Clear mask if confounder is random
      confounded = img.copy() # To remove conf patch
      mask = np.zeros_like(img)

    xc.append(confounded)
    masks.append(mask)

  return np.array(xc), np.array(masks)


class DecoyDataset(Dataset):
  """Confounded dataset."""

  def __init__(
    self,
    indices: np.ndarray,
    x: np.ndarray, 
    y: np.ndarray, 
    masks: np.ndarray, 
    transform: Optional[transforms.Compose]=None):
    """Initialize DecoyDataset.
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
    # Map from original MNIST id to positional id
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
  

  

def prepare_generic_data(
  dataset_class: Callable,
  raw_dir: str,
  save_path:str,
  val_size: float=0.2, 
  random_state: int=123,
  bias_ratio: list=[1]*N_CLASSES,
  variation: int = 0
) -> None:
  """Prepare data by loading and confounding it.
  Args:
    val_size (float): how big to make val set with reference to train set.
    random_state (int): random state for reproducibility.
  """
  os.makedirs(raw_dir, exist_ok=True)
  raw_train = dataset_class(raw_dir, train=True, download=True)
  raw_test = dataset_class(raw_dir, train=False, download=True)

  # Indices to track samples
  train_val_indices = np.arange(len(raw_train))
  test_indices = np.arange(len(raw_test))

  # Split train in train and val set
  train_idx, val_idx = train_test_split(
    train_val_indices,
    test_size=val_size,
    random_state=random_state,
    stratify=raw_train.targets
  )

  # Convert all datasets to numpy
  x_train = raw_train.data.numpy()[train_idx]
  y_train = raw_train.targets.numpy()[train_idx]
  x_val = raw_train.data.numpy()[val_idx]
  y_val = raw_train.targets.numpy()[val_idx]
  x_test = raw_test.data.numpy()
  y_test = raw_test.targets.numpy()

  # Confound data
  x_train_c, mask_train = confound_dataset(
    x_train, 
    y_train, 
    bias_ratio=bias_ratio, 
    variation=variation)
  x_val_c, mask_val   = confound_dataset(
    x_val, 
    y_val, 
    bias_ratio=[0]*N_CLASSES,
    variation=variation)
  x_test_c, mask_test  = confound_dataset(
    x_test, 
    y_test, 
    bias_ratio=[0]*N_CLASSES,
    variation=variation)

  # Save prepared data
  os.makedirs(save_path, exist_ok=True)
  np.savez_compressed(os.path.join(save_path, "train.npz"), x=x_train_c, y=y_train, mask=mask_train, indices=train_idx)
  np.savez_compressed(os.path.join(save_path,"val.npz"), x=x_val_c, y=y_val, mask=mask_val, indices=val_idx)
  np.savez_compressed(os.path.join(save_path,"test.npz"), x=x_test_c, y=y_test, mask=mask_test, indices=test_indices)


def dataset_mean_std(x_data: np.ndarray) -> tuple:
  # Scale to [0, 1]
  data = x_data / 255.0
  mean = data.mean()
  std = data.std()
  return (mean,), (std,)



def load_decoy(
    save_path: str,
    prepare_fn: Callable,
    seed: int = 123, 
    reload: bool = False, 
    bias_ratio: list = [1]*N_CLASSES,
    variation: int = 0
  ) -> tuple:
  """Load decoy dataset.
  Args:
    seed (int): seed for reproducibility.
    reload (bool): flag to reload dataset.
  Returns:
    tuple: train, val and test set.
  """
  # If possible, load data from "dataset/data/decoy_mnist" else recreate it
  train_path = os.path.join(save_path, "train.npz")
  val_path = os.path.join(save_path, "val.npz")
  test_path = os.path.join(save_path,"test.npz")

  files_exist = os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)
  if not files_exist or reload:
    prepare_fn(random_state=seed, bias_ratio=bias_ratio, variation=variation)

  # Load from .npz
  train_np = np.load(train_path)
  val_np = np.load(val_path)
  test_np = np.load(test_path)

  mean, std = dataset_mean_std(train_np["x"])

  # Transform that normalizes
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
  ])

  train_ds = DecoyDataset(train_np["indices"], train_np["x"], train_np["y"], train_np["mask"], transform)
  val_ds = DecoyDataset(val_np["indices"], val_np["x"],   val_np["y"],   val_np["mask"], transform)
  test_ds = DecoyDataset(test_np["indices"], test_np["x"],  test_np["y"],  test_np["mask"], transform)

  return train_ds, val_ds, test_ds
