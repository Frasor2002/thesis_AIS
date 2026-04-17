import torch
import numpy as np
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint, load_checkpoint
from functions.xai import explain_dataset
from functions.xil import compute_simplicity
from utils.utils import enable_reproducibility
from experiments.utils import compute_correlations

def compute_exp_entropy(attrs: torch.Tensor):
  """Compute entropy from explainations."""

  # Absolute value
  abs_attrs = torch.abs(attrs)

  # Flatten to easily sum per image 
  # (B, C, H, W) -> (B, C * H * W)
  flat_attrs = abs_attrs.flatten(start_dim=1)

  # Sum attributions for each image
  sum_attrs = torch.sum(flat_attrs, dim=1, keepdim=True)

  # Normalize to create a probability distribution (sum of 1)
  # Small epsilon to avoid division by zero
  epsilon = 1e-12
  probs = flat_attrs / (epsilon + sum_attrs)

  # Compute Shannon entropy
  # Epsilon to avoid log(0) = -inf
  entropy = -torch.sum(probs * torch.log2(probs + epsilon), dim=1)

  return entropy

def exp_explaination_entropy(
    seed: int = 123, 
    model_name: str = "LeNet",
    model_variant: str = "modern",
    dataset: str = "DecoyMNIST",
    variation: int = 0) -> dict:
  """Run experiment to see how explaination entropy correlate with confounder presence.
  Args:
    seed (int): seed for the experiment.
    dataset (str): dataset to use for the experiment.
    metric (str): metric to use for simplicity.
  Returns:
    dict: correlation between confounder presence and explaination entropy.
  """
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'

  enable_reproducibility(seed)
  model = load_model(model_name, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2, weight_decay=0)
  loss = load_loss_fun("CrossEntropy")

  train_set, val_set, test_set = load_data(
    dataset, 
    seed=seed, 
    reload=True,
    bias_ratio=[0.99]*10,
    variation=variation
  )
  
  data = [train_set, val_set, test_set]
  params = {"batch_size":32}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  _, _ = train_model(
    model, 
    train_loader, 
    optim, 
    loss, 
    n_epochs=10, 
    eval_loader=val_loader, 
    device=device
  )

  all_attr, all_images = explain_dataset(train_loader, model, device)

  entropy = compute_exp_entropy(all_attr)

  separation_list = []
  is_confounded = []
  labels = []
  for id in range(len(train_set)):
    _, _, label, mask = train_set[id] 
    
    # score -> 1 means confounder presence
    separation_score = -entropy[id].item()
    
    separation_list.append(separation_score)
    is_confounded.append(1 if mask.sum() > 1 else 0)
    labels.append(label.item())


  result = compute_correlations(separation_list, is_confounded, labels)

  return result