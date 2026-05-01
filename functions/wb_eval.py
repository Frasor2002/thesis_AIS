from typing import Callable, Tuple
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

def wb_eval(model: nn.Module, eval_loader: DataLoader, loss_fun: Callable, device: str = "cpu") -> Tuple[float, float, float, dict]:
  """Eval model on all eval data and compute worst group accuracy.
  Args:
    model (Module): pytorch model to evaluate.
    eval_loader (DataLoader): eval dataloader.
    loss_fun (Callable): loss function.
    device (str): device where to perform evaluation.
  Returns:
    tuple: evaluation loss, overall accuracy, worst group accuracy, and a dictionary of all group accuracies.
  """
  
  running_loss = 0.0
  correct_preds = 0
  tot_samples = 0
  
  group_correct = {}
  group_total = {}
  
  dataset = eval_loader.dataset
  
  model.eval()
  loop = tqdm(eval_loader, desc="Evaluating", leave=False)

  with torch.no_grad():
    for indeces, imgs, targets, masks in loop:
      imgs, targets = imgs.to(device), targets.to(device)
      logits = model(imgs)

      # Accumulate number of samples
      batch_size = imgs.size(0)
      tot_samples += batch_size

      # Loss
      loss = loss_fun(logits, targets)
      running_loss += loss.item() * batch_size

      # Accuracy
      _, predicted_labels = torch.max(logits, 1)
      correct_preds += (predicted_labels == targets).sum().item()
      
      # WORST GROUP ACCURACY
      preds_cpu = predicted_labels.cpu().numpy()
      targets_cpu = targets.cpu().numpy()
      indeces_cpu = indeces.cpu().numpy()
      
      for i in range(batch_size):
          y = targets_cpu[i]
          orig_idx = indeces_cpu[i]
          
          # Fetch the place using the original ID (0 for land, 1 for water)
          p = dataset.get_place(orig_idx)
          
          group_key = f"y={y}_p={p}"
          
          # Initialize group if not present
          if group_key not in group_total:
              group_total[group_key] = 0
              group_correct[group_key] = 0
              
          group_total[group_key] += 1
          if preds_cpu[i] == y:
              group_correct[group_key] += 1

  # Compute overall averages
  eval_loss = running_loss / tot_samples
  eval_acc = correct_preds / tot_samples
  
  # Compute specific group accuracies
  group_accs = {}
  for group, total in group_total.items():
      group_accs[group] = group_correct[group] / total
      
  # Identify the worst performing group
  worst_group_acc = min(group_accs.values()) if group_accs else 0.0

  return eval_loss, eval_acc, worst_group_acc, group_accs