from typing import Callable, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

FUNCTION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(FUNCTION_DIR)
LOG_DIR = os.path.join(PROJ_DIR,"log","wb_log")

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: Optimizer, loss_fun: Callable, device: str = "cpu") -> tuple:
  """Train model on all training data.
  Args:
    model (Module): pytorch model to train.
    train_loader (DataLoader): training dataloader.
    optimizer (Optimizer): optimizer.
    loss_fun (Callable): loss function.
    device (str): device where to perform training.
  Returns:
    tuple: training loss, accuracy and per-sample training dynamics.
  """
  running_loss = 0.0
  running_ce_loss = 0.0
  running_rr_loss = 0.0
  correct_preds = 0
  tot_samples = 0
  epoch_dynamics = {}

  model.train()
  loop = tqdm(train_loader, desc="Training", leave=False)

  for indeces, imgs, targets, masks in loop:
    imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)

    imgs.requires_grad_(True)
    optimizer.zero_grad()
    logits = model(imgs)
    loss = loss_fun(logits, targets, imgs, masks)
    ce_val = getattr(loss_fun, "curr_ce", loss.item())
    rr_val = getattr(loss_fun, "curr_rr", 0.0)

    loss.backward()
    optimizer.step()

    # Training dynamics
    # Confidence scores
    probs = torch.softmax(logits, dim=1)
    true_class_confidences = probs.gather(1, targets.view(-1, 1)).squeeze(1)
    # Prediction
    _, predicted_labels = torch.max(logits, 1)
    # Correctness
    is_correct_list = (predicted_labels == targets).long()

    # Store in dictionary
    for i, idx in enumerate(indeces):
      idx = idx.item()
      epoch_dynamics[idx] = {
        "confidence": true_class_confidences[i].item(),
        "prediction": predicted_labels[i].item(),
        "correct": is_correct_list[i].item(),
        "target": targets[i].item()
      }

    # Compute training metrics
    batch_size = imgs.size(0)
    tot_samples += batch_size
    running_ce_loss += ce_val * batch_size
    running_rr_loss += rr_val * batch_size

    # Accuracy
    _, predicted_labels = torch.max(logits, 1)
    correct_preds += (predicted_labels == targets).sum().item()
  
  # Compute averages over entire data
  train_acc = correct_preds / tot_samples
  train_ce_loss = running_ce_loss / tot_samples
  train_rr_loss = running_rr_loss / tot_samples

  return train_ce_loss, train_rr_loss, train_acc, epoch_dynamics


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


def wb_train(model: nn.Module, train_loader: DataLoader, optimizer: Optimizer, loss_fun: Callable, 
             n_epochs: int, eval_loader: Optional[DataLoader]=None, eval_loss_fun: Optional[Callable]=None,
             scheduler: Optional[Callable] = None, patience: int = 0, device: str = "cpu") -> tuple:
  """Train model for a number of epochs and log group accuracies using wb_eval.
  
  Args:
    model (Module): model to train.
    train_loader (DataLoader): training dataloader.
    optimizer (Optimizer): optimizer for training.
    loss_fun (Callable): loss function to train.
    n_epochs (int): number of epochs.
    eval_loader (Optional[DataLoader]): optional val dataloader.
    eval_loss_fun (Optional[Callable]): loss function for evaluation.
    scheduler (Optional[Callable]): optional lr scheduler.
    patience (int): early stopping patience.
    device (str): device where to compute tensors.
      
  Returns:
    tuple: training log for every epoch (including groups) and training dynamics.
  """
  log = {
    "epoch": [],
    "train_ce_loss": [],
    "train_rr_loss": [],
    "train_acc": [],
    "eval_loss": [],
    "eval_acc": [],
    "worst_group_acc": [],
    "group_accs": {} # Dictionary to store lists of accuracies per group
  }
  
  training_dynamics = {}
  best_val_loss = float('inf')
  patience_counter = 0
  loop = tqdm(range(n_epochs), desc="Training model")

  # Fallback if no separate eval loss function is provided
  if eval_loss_fun is None:
    eval_loss_fun = loss_fun

  for epoch in loop:
    loop.set_description(f"Epoch {epoch + 1}/{n_epochs}")

    # Train for one epoch using your existing train_epoch function
    train_ce_loss, train_rr_loss, train_acc, epoch_dyn = train_epoch(model, train_loader, optimizer, loss_fun, device)

    for sample_id, metrics in epoch_dyn.items():
      if sample_id not in training_dynamics:
        training_dynamics[sample_id] = []
      metrics["epoch"] = epoch
      training_dynamics[sample_id].append(metrics)

    # Log training metrics
    log["epoch"].append(epoch + 1)
    log["train_ce_loss"].append(train_ce_loss)
    log["train_rr_loss"].append(train_rr_loss)
    log["train_acc"].append(train_acc)

    info_dict = {
      "ce_loss": f"{train_ce_loss:.4f}",
      "acc": f"{train_acc:.4f}"
    }

    # Evaluate model using your wb_eval function
    if eval_loader is not None:
      eval_loss, eval_acc, worst_group_acc, group_accs = wb_eval(model, eval_loader, eval_loss_fun, device)
      
      # Log standard eval metrics
      log["eval_loss"].append(eval_loss)
      log["eval_acc"].append(eval_acc)
      log["worst_group_acc"].append(worst_group_acc)
      
      # Dynamically log each group's accuracy
      for group_key, acc in group_accs.items():
        if group_key not in log["group_accs"]:
          log["group_accs"][group_key] = []
        log["group_accs"][group_key].append(acc)

      info_dict["val_loss"] = f"{eval_loss:.4f}"
      info_dict["val_acc"] = f"{eval_acc:.4f}"
      info_dict["worst_grp"] = f"{worst_group_acc:.4f}"

      # Early stopping logic
      if patience > 0:
        if eval_loss < best_val_loss:
          best_val_loss = eval_loss
          patience_counter = 0  
        else:
          patience_counter += 1
          if patience_counter >= patience:
            loop.set_postfix(info_dict)
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

    if scheduler:
      scheduler.step()
    
    loop.set_postfix(info_dict)

  return log, training_dynamics


def wb_log_plot(log: dict, filename) -> None:
  """Plots training and evaluation logs, including worst group and specific group accuracies.
  
  Args:
    log (dict): Dictionary containing logged metrics from wb_train.
  """
  epochs = log["epoch"]
  
  # Setup the figure with 3 subplots side-by-side
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))
  
  # ---------------------------
  # Plot 1: Losses
  # ---------------------------
  axes[0].plot(epochs, log["train_ce_loss"], label="Train CE Loss", color="blue", marker="o", markersize=4)
  if log.get("eval_loss"):
    axes[0].plot(epochs, log["eval_loss"], label="Eval Loss", color="red", marker="s", markersize=4)
  
  axes[0].set_title("Cross Entropy Loss Over Epochs")
  axes[0].set_xlabel("Epoch")
  axes[0].set_ylabel("Loss")
  axes[0].grid(True, linestyle="--", alpha=0.6)
  axes[0].legend()

  # ---------------------------
  # Plot 2: Overall & Worst Group Accuracies
  # ---------------------------
  axes[1].plot(epochs, log["train_acc"], label="Train Acc", color="blue", marker="o", markersize=4)
  if log.get("eval_acc"):
    axes[1].plot(epochs, log["eval_acc"], label="Eval Acc", color="red", marker="s", markersize=4)
  if log.get("worst_group_acc"):
    axes[1].plot(epochs, log["worst_group_acc"], label="Worst Group Acc (Eval)", color="purple", linestyle="--", marker="x")

  axes[1].set_title("Overall & Worst Group Accuracy")
  axes[1].set_xlabel("Epoch")
  axes[1].set_ylabel("Accuracy")
  axes[1].grid(True, linestyle="--", alpha=0.6)
  axes[1].legend()

  # ---------------------------
  # Plot 3: Individual Group Accuracies
  # ---------------------------
  if log.get("group_accs"):
    for group_key, accs in log["group_accs"].items():
      axes[2].plot(epochs, accs, label=group_key, marker=".")
        
    axes[2].set_title("Individual Group Accuracies (Eval)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    # Place legend outside if there are many groups
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
  else:
    axes[2].text(0.5, 0.5, 'No group data logged', horizontalalignment='center', verticalalignment='center')
    axes[2].set_title("Individual Group Accuracies")

  plt.tight_layout()
  os.makedirs(LOG_DIR, exist_ok=True)
  path = os.path.join(LOG_DIR, f"{filename}.pdf")

  plt.savefig(path)