import numpy as np
from collections import defaultdict
from typing import Any, Set
import torch
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, load_checkpoint
from utils.utils import enable_reproducibility
from functions.xil import compute_simplicity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def get_confounded_classes(sampling_pool: list, simplicity: dict, dataset: Any, max_minority_ratio: float = 0.015, min_gap: float = 0.30) -> Set[int]:
    """
    Detects confounded classes by looking for an overwhelmingly dominant cluster
    (Simplicity near 0.99) leaving only a tiny fraction (< 1.5%) of unbiased samples.
    """
    class_simplicity = defaultdict(list)
    
    # 1. Group simplicity scores by class
    for internal_idx in sampling_pool:
        unique_id, _, y, _ = dataset[internal_idx]
        class_simplicity[int(y)].append(simplicity[unique_id])
        
    valid_classes = set()
    
    for cls, scores in class_simplicity.items():
        scores_arr = np.array(scores).reshape(-1, 1)
        
        if len(scores_arr) < 2:
            continue
            
        # 2. Force a 2-way split
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scores_arr)
        
        # 3. Calculate Gap
        centers = kmeans.cluster_centers_.flatten()
        gap = abs(centers[0] - centers[1])
        
        # 4. Calculate Minority Ratio
        labels = kmeans.labels_
        cluster_counts = np.bincount(labels)
        minority_ratio = np.min(cluster_counts) / len(labels)
        
        print(f"Class {cls}: Gap = {gap:.4f} | Minority Ratio = {minority_ratio:.4f} | Centers = [{centers[0]:.2f}, {centers[1]:.2f}]")
        
        # 5. Flip the logic: Confounded classes have a TINY minority of clean samples
        if minority_ratio < max_minority_ratio and gap > min_gap:
            valid_classes.add(cls)
            
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
    
  confounded_classes = get_confounded_classes(
    sampling_pool=sampling_pool, 
    simplicity=simplicity, 
    dataset=train_set,
    max_minority_ratio=0.03,
    min_gap=0.30
  )
    
  print("="*20, f"Detected Confounded Classes: {confounded_classes}", "="*20)
    
  return confounded_classes

