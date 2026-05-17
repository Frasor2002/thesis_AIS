import os
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model
from functions.wb_eval import wb_train
from functions.chc_eval import celeba_train
from functions.xai import explain_dataset
from functions.loss import load_loss_fun
import torch
import random


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SALIENCY_PATH = os.path.join(CURR_DIR, "saliency")

def prepare_mnist_saliency(seed, device, dataset):
  model = load_model("ModernLeNet", device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2)
  loss = load_loss_fun("CrossEntropy")
  
  train_set, val_set, test_set = load_data(
    dataset, 
    seed=seed, 
    reload=True,
    bias_ratio=[0.99]*10,
    variation=2,
    train_patch=False
  )
  data = [train_set, val_set, test_set]
  params = {"batch_size":32}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  log, _ = train_model(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss, 
    n_epochs=10, 
    eval_loader=val_loader, 
    device=device
  )
  all_attr, _ = explain_dataset(train_loader, model, device)
  os.makedirs(SALIENCY_PATH, exist_ok=True)
  saliency_dict = {}

  for i in range(len(train_set)):
    item_id = train_set[i][0]
    saliency_map = all_attr[i]
    saliency_dict[item_id] = saliency_map
  
  save_path = os.path.join(SALIENCY_PATH, f"{dataset}_sal.pth")
  torch.save(saliency_dict, save_path)
  print(f"Successfully saved {len(saliency_dict)} saliency maps to '{save_path}'")


def load_mnist_saliency(seed, device, dataset):
  load_path = os.path.join(SALIENCY_PATH, f"{dataset}_sal.pth")
  if not os.path.exists(load_path):
    prepare_mnist_saliency(seed, device, dataset)
    
  saliency_dict = torch.load(load_path, map_location=device, weights_only=False)
  print(f"Loaded {len(saliency_dict)} saliency maps from '{load_path}'")
    
  return saliency_dict



def prepare_wb_saliency(seed, device):
  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2)
  loss = load_loss_fun("CrossEntropy")
  train_set, val_set, test_set = load_data("Waterbirds", reload=False, balance=True, seed=seed)
  data = [train_set, val_set, test_set]
  params = {"batch_size":64}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)
  _, _ = wb_train(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss, 
    n_epochs=60,
    eval_loader=val_loader, 
    patience=3,
    device=device
  )
  all_attr, _ = explain_dataset(train_loader, model, device)
  os.makedirs(SALIENCY_PATH, exist_ok=True)
  saliency_dict = {}

  for i in range(len(train_set)):
    item_id = train_set[i][0]
    saliency_map = all_attr[i]
    saliency_dict[item_id] = saliency_map
  
  save_path = os.path.join(SALIENCY_PATH, "wb_sal.pth")
  torch.save(saliency_dict, save_path)
  print(f"Successfully saved {len(saliency_dict)} saliency maps to '{save_path}'")

def load_wb_saliency(seed, device="cuda"):
  load_path = os.path.join(SALIENCY_PATH, "wb_sal.pth")
  if not os.path.exists(load_path):
    prepare_wb_saliency(seed, device)
    
  saliency_dict = torch.load(load_path, map_location=device, weights_only=False)
  print(f"Loaded {len(saliency_dict)} saliency maps from '{load_path}'")
    
  return saliency_dict


def prepare_celeba_saliency(seed, device):
  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-3)
  loss = load_loss_fun("CrossEntropy")
  train_set, val_set, test_set = load_data("CelebAHairColor", reload=False, seed=seed)
  data = [train_set, val_set, test_set]
  params = {"batch_size": 64}
  m_params = [params] * 3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)
  
  # Ensure celeba_train is imported 
  _, _ = celeba_train(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss, 
    n_epochs=50, 
    eval_loader=val_loader, 
    patience=3,
    device=device
  )
  
  all_attr, _ = explain_dataset(train_loader, model, device)
  os.makedirs(SALIENCY_PATH, exist_ok=True)
  saliency_dict = {}

  for i in range(len(train_set)):
    item_id = train_set[i][0]
    saliency_map = all_attr[i]
    saliency_dict[item_id] = saliency_map
  
  save_path = os.path.join(SALIENCY_PATH, "celeba_sal.pth")
  torch.save(saliency_dict, save_path)
  print(f"Successfully saved {len(saliency_dict)} saliency maps to '{save_path}'")

def load_celeba_saliency(seed, device="cuda"):
  load_path = os.path.join(SALIENCY_PATH, "celeba_sal.pth")
  if not os.path.exists(load_path):
    prepare_celeba_saliency(seed, device)
    
  saliency_dict = torch.load(load_path, map_location=device, weights_only=False)
  print(f"Loaded {len(saliency_dict)} saliency maps from '{load_path}'")
    
  return saliency_dict



def saliency_sampler(dataset, saliency_dict, n_classes, k):
  if k % 2 != 0:
    raise ValueError("K must be even for sampling.")

  half_k = k // 2
  
  # For each class split conf and not conf
  buckets = {y: {0: [], 1: []} for y in range(n_classes)}
  
  # Group samples by class
  for i in range(len(dataset)):
    item_id, img, label, mask = dataset[i]
    
    # Find if confounded (alternative depend on dataset)
    is_confounded = 1 if mask.sum().item() > 1 else 0
    y = label.item() if isinstance(label, torch.Tensor) else label
    
    buckets[y][is_confounded].append(i)
    
  sampled_data = []
  
  # Sample from each bucket
  for y in range(n_classes):
    for c in [0, 1]:
      group_indices = buckets[y][c]
      
      # Shuffle to ensure random sampling
      random.shuffle(group_indices)
      sampled_indices = group_indices[:half_k]
      
      if len(sampled_indices) < half_k:
        print(f"Warning: Only found {len(sampled_indices)} samples for class {y}, confounded={c} (requested {half_k}).")
        
      # Retrieve img and saliency
      for idx in sampled_indices:
        item_id, img, label, mask = dataset[idx]
        
        if item_id in saliency_dict:
          sampled_data.append({
            "id": item_id,
            "img": img,
            "label": y,
            "mask": mask,
            "confounded": c,
            "saliency": saliency_dict[item_id]
          })
        else:
          print(f"Warning: Item {item_id} not found in saliency_dict. Skipping.")
          
  return sampled_data