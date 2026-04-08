import torch
import numpy as np
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import eval_model, save_checkpoint, load_checkpoint
from functions.xai import explain_dataset, evaluate_explainations
from utils.utils import enable_reproducibility
from typing import Callable, Tuple
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
from functions.loss import load_loss_fun
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from experiments.utils import compute_correlations, compute_auc_roc,log_corr_results, log_auc_results
from functions.xil import compute_simplicity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "log")
PLOT_DIR = os.path.join(LOG_DIR, "plot")

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
  correct_preds = 0
  tot_samples = 0

  confounded_loss_total = 0.0
  confounded_count = 0
  unconfounded_loss_total = 0.0
  unconfounded_count = 0

  epoch_dynamics = {}


  model.train()
  loop = tqdm(train_loader, desc="Training", leave=False)

  for indeces, imgs, targets, masks in loop:
    imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)

    imgs.requires_grad_(True)
    optimizer.zero_grad()
    logits = model(imgs)
    loss = loss_fun(logits, targets, imgs, masks)
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
    running_loss += loss.item() * batch_size
    # Accuracy
    _, predicted_labels = torch.max(logits, 1)
    correct_preds += (predicted_labels == targets).sum().item()

    with torch.no_grad():
      is_confounded = masks.view(batch_size, -1).sum(dim=1) > 0   
      per_sample_loss = F.cross_entropy(logits, targets, reduction='none')
      confounded_loss_total += per_sample_loss[is_confounded].sum().item()
      confounded_count += is_confounded.sum().item()
      unconfounded_loss_total += per_sample_loss[~is_confounded].sum().item()
      unconfounded_count += (~is_confounded).sum().item()
  
  # Compute averages over entire data
  train_loss = running_loss / tot_samples
  train_acc = correct_preds / tot_samples
  avg_confounded_loss = confounded_loss_total / confounded_count if confounded_count > 0 else 0.0
  avg_unconfounded_loss = unconfounded_loss_total / unconfounded_count if unconfounded_count > 0 else 0.0

  return train_loss, train_acc, avg_confounded_loss, avg_unconfounded_loss, epoch_dynamics




def train_model(model: nn.Module, train_loader: DataLoader, optimizer: Optimizer, loss_fun: Callable, n_epochs: int, eval_loader: Optional[DataLoader]=None, scheduler: Optional[Callable] = None, device:str="cpu") -> tuple:
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
    "train_loss": [],
    "train_acc": [],
    "train_confounded_loss": [],
    "train_unconfounded_loss": [],
    "eval_loss": [],
    "eval_acc": []
  }
  training_dynamics = {}

  loop = tqdm(range(n_epochs), desc="Training model")

  for epoch in loop:
    loop.set_description(f"Epoch {epoch + 1}/{n_epochs}")

    # Train for one epoch
    train_loss, train_acc, avg_confounded_loss, avg_unconfounded_loss, epoch_dyn = train_epoch(model, train_loader, optimizer, loss_fun, device)

    for sample_id, metrics in epoch_dyn.items():
      # init dictionary
      if sample_id not in training_dynamics:
        training_dynamics[sample_id] = []
      # Update training dynamics
      metrics["epoch"] = epoch
      training_dynamics[sample_id].append(metrics)

    log["epoch"].append(epoch)
    log["train_loss"].append(train_loss)
    log["train_acc"].append(train_acc)
    log["train_confounded_loss"].append(avg_confounded_loss)
    log["train_unconfounded_loss"].append(avg_unconfounded_loss)

    info_dict = {
      "loss": f"{train_loss:.4f}",
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

    if scheduler:
      scheduler.step()
    
    loop.set_postfix(info_dict)
  
  return log, training_dynamics


def plot_training_log(log: dict, filename: str) -> None:
    """Plots the training metrics over epochs.
    Args:
      log (dict): Dictionary containing the training log metrics.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    filepath1 = os.path.join(PLOT_DIR, f"{filename}_train.pdf")
    filepath2 = os.path.join(PLOT_DIR, f"{filename}_conf.pdf")
    epochs = [e + 1 for e in log["epoch"]] 

    fig1, axes = plt.subplots(1, 2, figsize=(18, 5))

    axes[0].plot(epochs, log["train_loss"], label="Train Loss", marker='o')
    if log["eval_loss"]:
        axes[0].plot(epochs, log["eval_loss"], label="Eval Loss", marker='s')
    axes[0].set_title("Overall Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, log["train_acc"], label="Train Acc", marker='o')
    if log["eval_acc"]:
        axes[1].plot(epochs, log["eval_acc"], label="Eval Acc", marker='s')
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    fig1.tight_layout()
    fig1.savefig(filepath1)
    plt.close()


    fig2, ax = plt.subplots(figsize=(6, 5))
    ax.plot(epochs, log["train_confounded_loss"], label="Confounded Loss", marker='o', linestyle='--')
    ax.plot(epochs, log["train_unconfounded_loss"], label="Unconfounded Loss", marker='s', linestyle='-.')
    ax.set_title("Train Loss by Subgroup")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yticks(np.arange(0, 5, 0.5))
    ax.legend()
    ax.grid(True)
    
    fig2.tight_layout()
    fig2.savefig(filepath2)
    plt.close()


def exp_confounder_study(
  seed: int = 123, 
  model_name : str = "ModernLeNet",
  dataset: str = "DecoyMNIST",
  bias_ratio: list = [0.99]*10,
  conf_type: int = 0,
  add:str = ""
) -> None:
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
    variation=conf_type
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
  #avg_penalty, conf_ratio = eval_confounding(model, train_loader, device)
  #all_attr, _ = explain_dataset(train_loader, model, device)
  #exp_penalty, class_penalty = evaluate_explainations(all_attr, train_set.masks, train_set.y)
  #print("="*20,f"Exp penalty:{exp_penalty:.2f}.","="*20)
  #print(class_penalty)

  #print("="*20,f"Avg penalty:{avg_penalty:.2f} | Conf ratio:{conf_ratio:.2f}.","="*20)

  # Plot losses and training
  print(" | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in log.items()))
  plot_training_log(log, f"cs_{dataset}_{conf_type}_{model_name}_{add}")

  simplicity = compute_simplicity(dyn, metric="MP")

  separation_list = []
  is_confounded = []
  labels = []
  for id in range(len(train_set)):
    index, _, label, mask = train_set[id] 
    separation_list.append(simplicity[index])
    is_confounded.append(1 if mask.sum() > 1 else 0)
    labels.append(label.item())

  result = compute_correlations(separation_list, is_confounded, labels)
  log_corr_results(result, filename=f"td_corr_{model_name}_{dataset}_{conf_type}_{add}")
  result = compute_auc_roc(separation_list, is_confounded, labels)
  log_auc_results(result, filename=f"td_auroc_{model_name}_{dataset}_{conf_type}_{add}")