import torch
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint, load_checkpoint
from functions.xil import xil_loop, random_sampling, simplicity_sampling
from utils.utils import enable_reproducibility
from typing import Any

RESET_CHECKPOINT="reset_model"

def run_mnist_xil(
  seed: int = 123, 
  model_name : str = "ModernLeNet",
  bias_ratio: list = [0.99]*10,
  conf_type: int = 0,
  train_patch: bool= False,
  sampling_strategy: str = "random",
  budget:int=1000,
  step:int=1,
  initial_query:int=0,
):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model(model_name, device=device)
  # Same weights for all successive iterations
  load_checkpoint(RESET_CHECKPOINT, model, device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2, weight_decay=0)
  loss = load_loss_fun("CrossEntropy")

  train_set, val_set, test_set = load_data(
    name="DecoyMNIST", 
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

  _, dyn = train_model(
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

  # Run XIL loop
  log = xil_loop(
    train_data=train_set,
    model=model,
    lr=1e-3,
    epochs=10,
    sampling_strategy=sampling_strategy,
    budget=budget,
    val_loader=val_loader,
    test_loader=test_loader,
    tr_dynamics=dyn,
    step_size=step,
    starting_query=initial_query,
    rrr_reg_rate=1e2,
    log_filename=f"MNIST_{seed}_{sampling_strategy}_{bias_ratio}_{model_name}",
    device=device
  ) 
  return log


def run_fmnist_xil(
  seed: int = 123, 
  model_name : str = "ModernLeNet",
  bias_ratio: list = [0.99]*10,
  conf_type: int = 0,
  train_patch: bool= False,
  sampling_strategy: str = "random",
  budget:int=1000,
  step:int=1,
  initial_query:int=0,
):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model(model_name, device=device)
  # Same weights for all successive iterations
  load_checkpoint(RESET_CHECKPOINT, model, device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2, weight_decay=0)
  loss = load_loss_fun("CrossEntropy")

  train_set, val_set, test_set = load_data(
    name="DecoyFashionMNIST", 
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

  _, dyn = train_model(
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

  # Run XIL loop
  log = xil_loop(
    train_data=train_set,
    model=model,
    lr=1e-3,
    epochs=10,
    sampling_strategy=sampling_strategy,
    budget=budget,
    val_loader=val_loader,
    test_loader=test_loader,
    tr_dynamics=dyn,
    step_size=step,
    starting_query=initial_query,
    rrr_reg_rate=1e2,
    log_filename=f"FMNIST_{seed}_{sampling_strategy}_{str(bias_ratio)}_{model_name}",
    device=device
  ) 
  return log




def run_wb_xil(
  sampling_strategy:str,
  budget:int,
  step:int,
  initial_query:int,
  seed:int=123,
):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True, device=device)
  # Load weights for all successive iterations
  load_checkpoint(RESET_CHECKPOINT, model, device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-1, weight_decay=0)
  loss = load_loss_fun("CrossEntropy")

  train_set, val_set, test_set = load_data(name="Waterbirds", reload=False, balance=True, seed=seed)
  
  data = [train_set, val_set, test_set]
  params = {"batch_size":64}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  _, dyn = train_model(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss, 
    n_epochs=100, 
    patience=3,
    eval_loader=val_loader, 
    device=device
  )
  loss, acc = eval_model(model, test_loader, loss,  device)
  print("="*20,f"Test set Loss:{loss:.2f} | Acc:{acc:.2f}.","="*20)

  # Run XIL loop
  log = xil_loop(
    train_data=train_set,
    model=model, 
    lr=1e-1,
    epochs=100,
    patience=3,
    sampling_strategy=sampling_strategy,
    budget=budget,
    val_loader=val_loader,
    test_loader=test_loader,
    tr_dynamics=dyn,
    step_size=step,
    starting_query=initial_query,
    rrr_reg_rate=1e2,
    log_filename=f"{sampling_strategy}_Waterbirds_{seed}",
    device=device
  )
  return log
  
  
  