from torchvision import datasets
from typing import Optional
import os
from dataset.decoy_dataset import prepare_generic_data, load_decoy

# Get dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
SAVE_PATH = os.path.join(DATA_PATH, "decoy_fashion_mnist")
RAW_DIR = os.path.join(DATA_PATH, "raw")

def prepare_data(
  val_size: float=0.2, 
  random_state: int=123,
  bias_ratio:list=[1]*10,
  variation:int=0,
  train_patch:bool=False
) -> None:
  """Prepare data by loading MNIST and confounding it.
  Args:
    val_size (float): how big to make val set with reference to train set.
    random_state (int): random state for reproducibility.
  """
  prepare_generic_data(
    dataset_class=datasets.FashionMNIST,
    raw_dir=RAW_DIR,
    save_path=SAVE_PATH,
    val_size=val_size,
    random_state=random_state,
    bias_ratio=bias_ratio,
    variation=variation,
    train_patch=train_patch
  )


def load_decoyFashionMNIST(seed: int = 123, reload: bool = True, bias_ratio:list=[1]*10, variation:int=0, train_patch:bool=False) -> tuple:
  """Load DecoyFashionMNIST dataset.
  Args:
    seed (int): seed for reproducibility.
    reload (bool): flag to reload dataset.
  Returns:
    Tuple[DecoyFashionMNIST, DecoyFashionMNIST, DecoyFashionMNIST]: train, val and test set.
  """
  return load_decoy(
    save_path=SAVE_PATH,
    prepare_fn=prepare_data,
    seed=seed,
    reload=reload,
    bias_ratio=bias_ratio,
    variation=variation,
    train_patch=train_patch
  )