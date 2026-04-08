from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from torchvision import transforms
from typing import Callable, Optional, Tuple
import torch
from PIL import Image
import tqdm
import matplotlib.pyplot as plt

# TODO read metadata, load images, create labels, load masks for confounded images in training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
IMG_PATH = os.path.join(BASE_DIR, "data", "waterbirds", "waterbird_complete95_forest2water2")
SEGMENTATION_PATH= os.path.join(BASE_DIR, "data", "waterbirds", "segmentations")
METADATA_FILE = os.path.join(IMG_PATH, "metadata.csv")

 
    
class Waterbirds(Dataset):
  def __init__(
    self,
    indices: np.ndarray,
    x: np.ndarray, 
    y: np.ndarray, 
    masks: np.ndarray, 
    transform: Optional[transforms.Compose]=None
    ):
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


def load_waterbirds():
  train_id = []
  train_x = []
  train_y = []
  train_mask = []
  val_id = []
  val_x = []
  val_y = []
  val_mask = []
  test_id = []
  test_x = []
  test_y = []
  test_mask = []


  metadata = pd.read_csv(METADATA_FILE)
  #print(metadata.head(10))
  # From this you have the img id, filename to load img and mask, y, split and place info
  for row in metadata.itertuples(index=False): # VERY SLOW

    img_id = row.img_id
    img_filename = str(row.img_filename)
    #TODO fix filename error for masks
    # TODO filename not compatible for every os
    split = row.split

    # Load image and mask
    img_path = os.path.join(IMG_PATH, img_filename)
    mask_path = os.path.join(SEGMENTATION_PATH, img_filename)
    mask_path = mask_path.replace(".jpg", ".png")
    with Image.open(img_path) as img:
      img = img.convert("RGB")
      img = np.array(img)
    #img = Image.open(img_path)
    #mask = Image.open(mask_path)
    with Image.open(mask_path) as mask:
      mask = np.array(mask)
    
    if split == 0:
      # train
      train_id.append(img_id)
      train_x.append(img)
      train_y.append(row.y)
      train_mask.append(mask)
    elif split == 1:
      # val
      val_id.append(img_id)
      val_x.append(img)
      val_y.append(row.y)
      val_mask.append(mask)
    elif split == 2:
      #test
      test_id.append(img_id)
      test_x.append(img)
      test_y.append(row.y)
      test_mask.append(mask)
    else:
      raise ValueError()
  
  print(len(train_id), len(val_id), len(test_id))
  import matplotlib.pyplot as plt
  plt.imshow(train_x[0])
  plt.show()
  plt.imshow(train_mask[0])
  plt.show()



if __name__ == "__main__":
  load_waterbirds()