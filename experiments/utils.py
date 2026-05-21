from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from numpy.typing import ArrayLike
import os
import matplotlib.pyplot as plt
import torch
from torchmetrics.functional.classification import binary_auroc
from utils.utils import enable_reproducibility 
from model.model import load_model
from functions.functions import save_checkpoint


# Name of checkpoint to reset the model
RESET_CHECKPOINT="reset_model"

def create_common_checkpoint(seed: int, model_name: str, diff="", **kwargs) -> None:
  """Create a common weight checkpoint at the start of an experiment to ensure fair comparison."""
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model(model_name, device=device, **kwargs)
  # Same weights for all successive iterations
  save_checkpoint(RESET_CHECKPOINT + diff, model)


def compute_correlations(separation_list: ArrayLike, is_confounded: ArrayLike, labels: ArrayLike) -> dict:
  """Function to compute correlation between a separation strategy and actual confounder
  presence.
  Args:
    separation_list (ArrayLike): list that tells results of the separation method. 
    is_confounded (ArrayLike): list with gt results of confounded and not-confounded.
    labels (ArrayLike): list with the labels for each sample for class-wise correlation.
  Returns:
    dict: total and classwise correlation. 
  """

  separation_list = np.array(separation_list)
  is_confounded = np.array(is_confounded)
  labels = np.array(labels)
  
  total_corr = pearsonr(separation_list, is_confounded)

  # Class-wise correlation
  class_corr = {}
  unique_classes = np.unique(labels)

  for label in unique_classes:
    class_mask = (labels == label)
    c_scores = separation_list[class_mask]
    c_conf = is_confounded[class_mask]
    
    if len(np.unique(c_conf)) > 1:
      class_corr[int(label)] = pearsonr(c_scores, c_conf)
    else:
      class_corr[int(label)] = np.nan

  return {
    "total": total_corr,
    "class": class_corr
  }


def compute_auc_roc(separation_list: ArrayLike, is_confounded: ArrayLike, labels: ArrayLike) -> dict:
  """Function to compute correlation between a separation strategy and actual confounder
  presence.
  Args:
    separation_list (ArrayLike): list that tells results of the separation method. 
    is_confounded (ArrayLike): list with gt results of confounded and not-confounded.
    labels (ArrayLike): list with the labels for each sample for class-wise correlation.
  Returns:
    dict: total and classwise auc roc score. 
  """
  preds = torch.as_tensor(separation_list)
  target = torch.as_tensor(is_confounded)
  labels_tensor = torch.as_tensor(labels)
    
  total_auc = binary_auroc(preds, target).item()

  # Class-wise AUC-ROC
  class_auc = {}
  unique_classes = torch.unique(labels_tensor)

  for label in unique_classes:
    class_mask = (labels_tensor == label)
    c_scores = preds[class_mask]
    c_conf = target[class_mask]
      
    if len(torch.unique(c_conf)) > 1:
      class_auc[int(label.item())] = binary_auroc(c_scores, c_conf).item()
    else:
      class_auc[int(label.item())] = float('nan')

  return {
    "total": total_auc,
    "class": class_auc
  }


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "log")
PLOT_DIR = os.path.join(LOG_DIR, "plot")


def log_corr_results(result: dict, filename: str) -> None:
  path = os.path.join(LOG_DIR, f"{filename}.log")
  os.makedirs(LOG_DIR, exist_ok=True)

  with open(path, 'w', encoding='utf-8') as f:
    total_corr = result['total']
    class_corr = result['class']
    
    f.write(f"Total correlation: stat={total_corr[0]:.4f} | pval={total_corr[1]}\n\n")
    
    for key, val in class_corr.items():
      if isinstance(val, float):
        f.write(f"Class correlation for label {key}: stat=NaN | pval=NaN\n")
      else:
        f.write(f"Class correlation for label {key}: stat={val[0]:.4f} | pval={val[1]}\n")


def log_auc_results(result: dict, filename: str) -> None:
  path = os.path.join(LOG_DIR, f"{filename}.log")
  os.makedirs(LOG_DIR, exist_ok=True)

  with open(path, 'w', encoding='utf-8') as f:
    total_auc = result['total']
    class_auc = result['class']
    
    f.write(f"Total auc: {total_auc:.4f}")
    f.write("\n\n")
    for key, val in class_auc.items():
      f.write(f"Class auc for label {key}: {val:.4f}")
      f.write("\n")
      
