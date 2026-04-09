from torch.utils.data.dataloader import DataLoader
from typing import List
from dataset.decoy_mnist import load_decoyMNIST
from dataset.decoy_fmnist import load_decoyFashionMNIST
from dataset.waterbirds import load_waterbirds
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_k_samples(dataset: Any, label: int, k: int = 5) -> None:
  """Visualization helper to see k samples with a given label.
  Args:
    dataset (Any): dataset to visualize.
    label (int): label of which samples are shown.
    k (int): number of images to show. Defaults to 5.
  """
  fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
  
  if k == 1:
    axes = [axes]

  tr_indices = np.where(dataset.y == label)[0]  
  k_actual = min(k, len(tr_indices))
  if k_actual == 0:
    print(f"No samples found for label {label}.")
    return
  selected_indices = np.random.permutation(tr_indices)[:k_actual]

  for a, i in enumerate(selected_indices):
    _, x, y, _ = dataset[i]
    if isinstance(x, torch.Tensor):
      x = x.cpu().numpy()
        
    if x.min() < 0 or x.max() > 1:
      x = (x - x.min()) / (x.max() - x.min())

    if len(x.shape) == 3:
      if x.shape[0] == 1:
        img_to_plot = x.squeeze(0)
        axes[a].imshow(img_to_plot, cmap='gray')
      elif x.shape[0] == 3:
        img_to_plot = np.transpose(x, (1, 2, 0))
        axes[a].imshow(img_to_plot)
      else:
        raise ValueError(f"Unsupported channel size: {x.shape[0]}")
    else:
      axes[a].imshow(x, cmap='gray')
    
    axes[a].set_xticks([])
    axes[a].set_yticks([])
    
    for spine in axes[a].spines.values():
      spine.set_edgecolor('gray')
      spine.set_linewidth(0.5)

  for a in range(k_actual, k):
    axes[a].axis('off')

  plt.tight_layout()



def load_data(name: str, **kwargs: Any):
  """Load a dataset.
  Args:
    name (str): name of the dataset.
    kwargs (Any): additional dataset arguments.
  Returns:
    tuple: train, val and test datasets.
  """
  datasets = {
    "DecoyMNIST": load_decoyMNIST,
    "DecoyFashionMNIST": load_decoyFashionMNIST,
    "Waterbirds": load_waterbirds
  }
  
  if name not in datasets.keys():
    raise ValueError("Wrong dataset name.")
  
  loader = datasets[name]
  train, val, test = loader(**kwargs)
  return train, val, test


def create_dataloaders(data_list: list, params_list: List[dict]) -> tuple:
  """Create dataloaders given train val and test and a list of params for each one.
  Args:
    data_list (list): list of datasets.
    params_list (list):list of dataloader params for each dataset.
  Returns:
    tuple: dataloaders.
  """
  loader_list = []
  for data, params in zip(data_list, params_list):
    data = data
    loader_list.append(DataLoader(data, **params))
  
  return tuple(loader_list)