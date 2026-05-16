import torch
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import eval_model, load_checkpoint
from functions.xai import explain_dataset, evaluate_explainations
from utils.utils import enable_reproducibility
from functions.loss import load_loss_fun
import os
import numpy as np
from experiments.utils import compute_correlations, compute_auc_roc,log_corr_results, log_auc_results
from functions.xil import compute_simplicity
from experiments.data_study_utils import train_model, plot_training_log

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "log")
PLOT_DIR = os.path.join(LOG_DIR, "plot")



def run_mnist_study(
  seed: int = 123, 
  model_name : str = "ModernLeNet",
  dataset: str = "DecoyMNIST",
  bias_ratio: list = [0.99]*10,
  conf_type: int = 0,
  train_patch:bool=False,
  add:str = ""
) -> None:
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model(model_name, device=device)
  #RESET_CHECKPOINT="reset_model"
  #load_checkpoint(RESET_CHECKPOINT, model, device)
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





# For other datasets to load
def run_wb_study(seed):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2, weight_decay=0)
  loss = load_loss_fun("CrossEntropy")
  train_set, val_set, test_set = load_data("Waterbirds",reload=False, balance=True, seed=seed)
  data = [train_set, val_set, test_set]
  params = {"batch_size":64}
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

  # Plot losses and training
  print(" | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in log.items()))
  plot_training_log(log, f"cs_waterbirds_{seed})")

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
  log_corr_results(result, filename=f"td_corr_Waterbirds")
  result = compute_auc_roc(separation_list, is_confounded, labels)
  log_auc_results(result, filename=f"td_auroc_Waterbirds")


def run_celeba_study(seed: int = 123):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-3, weight_decay=0)
  loss_fun = load_loss_fun("CrossEntropy")
  
  train_set, val_set, test_set = load_data("CelebAHC", reload=False)
  
  data = [train_set, val_set, test_set]
  params = {"batch_size": 64}
  m_params = [params] * 3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  log, dyn = train_model(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss_fun, 
    n_epochs=50,
    eval_loader=val_loader, 
    device=device
  )
  
  test_loss, acc = eval_model(model, test_loader, loss_fun, device)
  print("="*20, f"Test set Loss:{test_loss:.2f} | Acc:{acc:.2f}.", "="*20)

  # Plot losses and training
  print(" | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in log.items()))
  plot_training_log(log, f"cs_celeba_{seed}")

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
  log_corr_results(result, filename=f"td_corr_CelebA")
  result = compute_auc_roc(separation_list, is_confounded, labels)
  log_auc_results(result, filename=f"td_auroc_CelebA")