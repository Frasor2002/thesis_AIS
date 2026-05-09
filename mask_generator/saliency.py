import os
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model
from functions.wb_eval import wb_train
from functions.xai import explain_dataset
from functions.loss import load_loss_fun
import torch


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
    prepare_wb_saliency(seed, device)
    
  saliency_dict = torch.load(load_path, map_location=device, weights_only=False)
  print(f"Loaded {len(saliency_dict)} saliency maps from '{load_path}'")
    
  return saliency_dict



def prepare_wb_saliency(seed, device):
  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-1)
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
    n_epochs=100, 
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