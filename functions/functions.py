from typing import Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
from functions.loss import load_loss_fun
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

FUNCTION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(FUNCTION_DIR)
MODEL_DIR = os.path.join(PROJ_DIR, "model")

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
    # Loss (assume loss compute is mean loss over the batch)
    #running_loss += loss.item() * batch_size
    running_ce_loss += ce_val * batch_size
    running_rr_loss += rr_val * batch_size

    # Accuracy
    _, predicted_labels = torch.max(logits, 1)
    correct_preds += (predicted_labels == targets).sum().item()
  
  # Compute averages over entire data
  train_acc = correct_preds / tot_samples
  train_ce_loss = running_ce_loss / tot_samples
  train_rr_loss = running_rr_loss / tot_samples

  return train_ce_loss, train_rr_loss, train_acc, epoch_dynamics,


def eval_model(model: nn.Module, eval_loader: DataLoader, loss_fun: Callable, device: str = "cpu") -> Tuple[float, float]:
  """Eval model on all eval data.
  Args:
    model (Module): pytorch model to evaluate.
    eval_loader (DataLoader): eval dataloader.
    optimizer (Optimizer): optimizer.
    loss_fun (Callable): loss function.
    device (str): device where to perform evaluation.
  Returns:
    tuple: evaluation loss and accuracy.
  """
  
  running_loss = 0.0
  correct_preds = 0
  tot_samples = 0
  
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
  
  # Compute averages over entire data
  eval_loss = running_loss / tot_samples
  eval_acc = correct_preds / tot_samples

  return eval_loss, eval_acc
  


def train_model(model: nn.Module, train_loader: DataLoader, optimizer: Optimizer, loss_fun: Callable, n_epochs: int, eval_loader: Optional[DataLoader]=None, scheduler: Optional[Callable] = None,patience:int=0, device:str="cpu") -> tuple:
  """Train model for a number of epochs.
  Args:
    model (Module): model to train.
    train_loader (DataLoader): training dataloader.
    optimizer (Optimizer): optimizer for training.
    loss_fun (Callable): loss function to train.
    n_epochs (int): number of epochs.
    eval_loader (Optional[Dataloader]): optional val dataloader.
    scheduler (Optional[Callable]): optional lr scheduler.
    device (str): device where to compute tensors.
  Returns:
    tuple: training log for every epoch and training dynamics for each samples.
  """
  
  log = {
    "epoch": [],
    "train_ce_loss": [],
    "train_rr_loss": [],
    "train_acc": [],
    "eval_loss": [],
    "eval_acc": []
  }
  training_dynamics = {}
  best_val_loss = float('inf')
  patience_counter = 0
  loop = tqdm(range(n_epochs), desc="Training model")

  for epoch in loop:
    loop.set_description(f"Epoch {epoch + 1}/{n_epochs}")

    # Train for one epoch
    train_ce_loss,train_rr_loss, train_acc, epoch_dyn = train_epoch(model, train_loader, optimizer, loss_fun, device)

    for sample_id, metrics in epoch_dyn.items():
      # init dictionary
      if sample_id not in training_dynamics:
        training_dynamics[sample_id] = []
      # Update training dynamics
      metrics["epoch"] = epoch
      training_dynamics[sample_id].append(metrics)

    log["epoch"].append(epoch)
    log["train_ce_loss"].append(train_ce_loss)
    log["train_rr_loss"].append(train_rr_loss)
    log["train_acc"].append(train_acc)

    info_dict = {
      "ce_loss": f"{train_ce_loss:.4f}",
      "rr_loss": f"{train_rr_loss:.4f}",
      "acc": f"{train_acc:.4f}"
    }

    # If eval is passed, eval the model
    if eval_loader is not None:
      eval_loss_fun = load_loss_fun("CrossEntropy")
      eval_loss, eval_acc = eval_model(model, eval_loader, eval_loss_fun, device)
      log["eval_loss"].append(eval_loss)
      log["eval_acc"].append(eval_acc)

      info_dict["val_loss"] = f"{eval_loss:.4f}"
      info_dict["val_acc"] = f"{eval_acc:.4f}"
      if patience > 0:
        if eval_loss < best_val_loss:
          best_val_loss = eval_loss
          patience_counter = 0  # Reset counter if validation loss improves
        else:
          patience_counter += 1
          if patience_counter >= patience:
            loop.set_postfix(info_dict)
            print(f"\nEarly stopping triggered after {epoch + 1} epochs! No improvement in validation loss for {patience} epochs.")
            break # Stop training

    if scheduler:
      scheduler.step()
    
    loop.set_postfix(info_dict)
  
  return log, training_dynamics


def save_checkpoint(model_name: str, model: nn.Module) -> None:
  """Save model checkpoint.
  Args:
    model_name (str): name given to the checkpoint.
    model (nn.Module): model to save a checkpoint of.
  """
  folder_path = os.path.join(MODEL_DIR,'bin')
  os.makedirs(folder_path, exist_ok=True)
  path = os.path.join(folder_path, model_name + '.pt')
  torch.save(model.state_dict(), path)


def load_checkpoint(model_name: str, model: nn.Module, device: str = "cpu") -> None:
  """Load model checkpoint.
  Args:
    model_name (str): name given to the checkpoint.
    model (nn.Module): model to load a checkpoint of.
    device (str): device where the model is located for correct loading.
  """
  path = os.path.join(MODEL_DIR, 'bin', model_name + '.pt')
  model.load_state_dict(torch.load(path, map_location=torch.device(device)))


def plot_training_dyn(metrics: list) -> None:
  confidence = [data['confidence'] for data in metrics]
  correctness = [data['correct'] for data in metrics]
  target = metrics[0]['target']
  epochs = [data['epoch'] + 1 for data in metrics]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
  ax1.plot(epochs, confidence, label=f'Target {target}')
  ax2.step(epochs, correctness, label=f'Target {target}')
  ax1.set_title('Confidence over Epochs')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('True Class Probability')
  ax1.set_xticks(epochs)
  ax1.legend()
  ax1.grid(True)

  ax2.set_title('Correctness over Epochs')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Correct (1) / Incorrect (0)')
  ax2.set_yticks([0, 1])
  ax2.set_xticks(epochs)
  ax2.legend()
  ax2.grid(True)
  plt.tight_layout()
  plt.show()


def visualize_5_sample_dynamics(training_dynamics: dict, label: int) -> None:
  target_indices = [uid for uid, metrics in training_dynamics.items() if metrics[0]['target'] == label]

  if not target_indices:
    print(f"No samples found for class {label}")
    return

  num_samples = min(5, len(target_indices))
  selected_indices = np.random.choice(target_indices, num_samples, replace=False)

  for sample_id in selected_indices:
    metrics = training_dynamics[sample_id]
    plot_training_dyn(metrics)