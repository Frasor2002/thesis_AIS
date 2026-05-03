import numpy as np
from collections import defaultdict
from typing import Any, Set
from sympy import plot
import torch
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, load_checkpoint
from utils.utils import enable_reproducibility
from functions.xil import compute_simplicity
from sklearn.cluster import KMeans, AgglomerativeClustering
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "log")
PLOT_DIR = os.path.join(LOG_DIR, "plot_class_selection")

def cls_kmeans_viz(class_data, max_minority, dataset_name, bs):   
  os.makedirs(PLOT_DIR, exist_ok=True)

  plt.figure(figsize=(10, 5))
  for cls, gap, minority_ratio in sorted(class_data, key=lambda x: x[0]):
    plt.scatter(minority_ratio, gap, label=f'Class {cls}', s=60)

  #plt.axhline(y=min_gap, color='red', linestyle='--', label=f'Gap > {min_gap}')
  plt.axvline(x=max_minority, color='blue', linestyle='--', label=f'Minority < {max_minority}')
  plt.xlabel('Minority Ratio')
  plt.ylabel('Gap')
  plt.title('Gap vs. Minority Ratio by Class')
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid(True)
  
  plt.tight_layout()
  path = os.path.join(PLOT_DIR, f"{dataset_name}_{bs}_plot.pdf")
  plt.savefig(path)

def get_confounded_classes(sampling_pool: list, simplicity: dict, dataset: Any, seed:int,data_name, bs, max_minority_ratio: float = 0.015) -> Set[int]:
  """
  Detects confounded classes by looking for an overwhelmingly dominant cluster
  (Simplicity near 0.99) leaving only a tiny fraction (< 1.5%) of unbiased samples.
  """
  class_simplicity = defaultdict(list)
  
  # Group simplicity by class
  for internal_idx in sampling_pool:
    unique_id, _, y, _ = dataset[internal_idx]
    class_simplicity[int(y)].append(simplicity[unique_id])
      
  valid_classes = set()
  plot_data = []
  
  for cls, scores in class_simplicity.items():
    scores_arr = np.array(scores).reshape(-1, 1)
      
    if len(scores_arr) < 2:
      continue
          
    # Apply KMEANS with k=2
    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(scores_arr)
    
    # Compute gap
    centers = kmeans.cluster_centers_.flatten()
    gap = abs(centers[0] - centers[1])
    
    # Minority ratio
    labels = kmeans.labels_
    cluster_counts = np.bincount(labels)
    minority_ratio = np.min(cluster_counts) / len(labels)
    plot_data.append((cls, gap, minority_ratio))
    
    print(f"Class {cls}: Gap = {gap:.4f} | Minority Ratio = {minority_ratio:.4f} | Centers = [{centers[0]:.2f}, {centers[1]:.2f}]")
    
    # Use thresholds
    if minority_ratio < max_minority_ratio: # gap < min_gap
      valid_classes.add(cls)
  
  cls_kmeans_viz(plot_data, max_minority_ratio, data_name, bs)

  return valid_classes


def run_class_selector(seed, model_name, dataset, bias_ratio, conf_type, train_patch):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model(model_name, device=device)
  RESET_CHECKPOINT="reset_model"
  load_checkpoint(RESET_CHECKPOINT, model, device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2, weight_decay=0)
  loss = load_loss_fun("CrossEntropy")
  train_set, val_set, test_set = load_data(
    dataset, 
    seed=seed, 
    reload=True,
    bias_ratio=bias_ratio,
    variation=conf_type,
    train_patch=train_patch
  )
  data = [train_set, val_set, test_set]
  params = {"batch_size":32}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  log, dyn = train_model(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss, 
    n_epochs=10, 
    eval_loader=val_loader,
    device=device
  )
  loss, acc = eval_model(model, test_loader, loss,  device)
  print("="*20,f"Test set Loss:{loss:.2f} | Acc:{acc:.2f}.","="*20)

  simplicity = compute_simplicity(dyn, metric="MP")

  sampling_pool = list(range(len(train_set)))

  if dataset == "DecoyMNIST": min_ratio = 0.0125
  else: min_ratio=0.03
    
  confounded_classes = get_confounded_classes(
    sampling_pool=sampling_pool, 
    simplicity=simplicity, 
    dataset=train_set,
    seed=seed,
    data_name=dataset,
    #silhouette_threshold=0.65,
    max_minority_ratio=min_ratio,
    bs=str(bias_ratio)
  )
    
  print("="*20, f"Detected Confounded Classes: {confounded_classes}", "="*20)
    
  return confounded_classes

