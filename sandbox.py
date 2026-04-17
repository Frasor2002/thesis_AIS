import torch
import numpy as np
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders, visualize_k_samples
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint, load_checkpoint
from functions.xai import explain_dataset, evaluate_explainations
from utils.utils import enable_reproducibility
from functions.loss import load_loss_fun
import matplotlib.pyplot as plt
import numpy as np

def mnist_test(seed, loss_name, dataset, bias_ratio, patch):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model("ModernLeNet", device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2)
  if loss_name == "CrossEntropy":
    loss = load_loss_fun(loss_name)
  else:
    loss = load_loss_fun("RRR", reg_rate = 1e2, normalize=True)
  
  train_set, val_set, test_set = load_data(
    dataset, 
    seed=seed, 
    reload=True,
    bias_ratio=bias_ratio,
    variation=2,
    train_patch=patch
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
  print(log)

  eval_loss = load_loss_fun("CrossEntropy")
  loss, acc = eval_model(model, test_loader, eval_loss, device)
  print("="*20,f"Test set Loss:{loss:.2f} | Acc:{acc:.2f}.","="*20)

if __name__ == "__main__":
  ce = "CrossEntropy"
  rrr = "RRR"
  mnist = "DecoyMNIST"
  fmnist = "DecoyFMNIST"
  mnist_test(123, rrr, mnist, [0.99]*10, True)