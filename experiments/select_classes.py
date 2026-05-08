import numpy as np
from collections import defaultdict
from typing import Any
import torch
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, load_checkpoint
from utils.utils import enable_reproducibility
from functions.xil import compute_simplicity
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "log")
PLOT_DIR = os.path.join(LOG_DIR, "plot_class_selection")

def cls_kmeans_viz(class_data, dataset_name, bs, max_minority=None):   
  os.makedirs(PLOT_DIR, exist_ok=True)

  plt.figure(figsize=(10, 5))
  for cls, gap, minority_ratio in sorted(class_data, key=lambda x: x[0]):
    plt.scatter(minority_ratio, gap, label=f'Class {cls}', s=60)

  # Make the threshold line optional since we are shifting to distributions
  if max_minority is not None:
    plt.axvline(x=max_minority, color='blue', linestyle='--', label=f'Minority < {max_minority}')
      
  plt.xlabel('Minority Ratio')
  plt.ylabel('Gap')
  plt.title('Gap vs. Minority Ratio by Class')
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid(True)
  
  plt.tight_layout()
  path = os.path.join(PLOT_DIR, f"{dataset_name}_{bs}_plot.pdf")
  plt.savefig(path)
  plt.close()

def plot_distribution(class_dist, dataset_name, bs):
  """
  Visualizes the final sampling probability distribution as a bar chart and saves it as a PDF.
  """
  os.makedirs(PLOT_DIR, exist_ok=True)
  
  classes = list(class_dist.keys())
  probabilities = list(class_dist.values())
  
  plt.figure(figsize=(10, 5))
  plt.bar(classes, probabilities, color='skyblue', edgecolor='black')
  
  plt.xlabel('Class')
  plt.ylabel('Sampling Probability')
  plt.title(f'Sampling Probability Distribution ({dataset_name}, bs={bs})')
  plt.xticks(classes)
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  
  plt.tight_layout()
  path = os.path.join(PLOT_DIR, f"{dataset_name}_{bs}_distribution_plot.pdf")
  plt.savefig(path)
  plt.close()


def get_sampling_distribution(sampling_pool: list, simplicity: dict, dataset: Any, seed:int, data_name, bs, temperature: float = 0.05) -> dict:
  """
  Computes minority ratios for each class and returns a probability distribution.
  Lower minority ratios yield higher sampling probabilities.
  """
  class_simplicity = defaultdict(list)
  
  # Group simplicity by class
  for internal_idx in sampling_pool:
    unique_id, _, y, _ = dataset[internal_idx]
    class_simplicity[int(y)].append(simplicity[unique_id])
      
  plot_data = []
  classes = []
  minority_ratios = []
  
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
    classes.append(cls)
    minority_ratios.append(minority_ratio)
    
    print(f"Class {cls}: Gap = {gap:.4f} | Minority Ratio = {minority_ratio:.4f} | Centers = [{centers[0]:.2f}, {centers[1]:.2f}]")

  # Visualize without the hard threshold line
  #cls_kmeans_viz(plot_data, data_name, bs)

  if not classes:
    return {}

  classes = np.array(classes)
  minority_ratios = np.array(minority_ratios)
  
  # Normalize the ratios to a [0, 1] scale
  min_r = np.min(minority_ratios)
  max_r = np.max(minority_ratios)
  
  if max_r > min_r:
    scaled_ratios = (minority_ratios - min_r) / (max_r - min_r)
  else:
    scaled_ratios = np.zeros_like(minority_ratios)
  
  neg_ratios = -scaled_ratios / temperature
  
  exp_ratios = np.exp(neg_ratios - np.max(neg_ratios))
  probabilities = exp_ratios / exp_ratios.sum()
  
  class_distribution = {int(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    
  
  return class_distribution


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

  if dataset == "DecoyMNIST": t = 0.1
  else: t=0.05
    
# Generate the probability distribution
  class_distribution = get_sampling_distribution(
    sampling_pool=sampling_pool, 
    simplicity=simplicity, 
    dataset=train_set,
    seed=seed,
    data_name=dataset,
    bs=str(bias_ratio),
    temperature=t 
  )
      
  print("="*20, f"Class Sampling Distribution:\n", class_distribution, "="*20)
  plot_distribution(class_distribution, dataset, str(bias_ratio))
      
  return class_distribution

"""
HOW TO USE
classes = list(class_dist.keys())
probabilities = list(class_dist.values())

sampled_class = np.random.choice(classes, p=probabilities)
"""