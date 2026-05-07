import torch
import numpy as np
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders, visualize_k_samples
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint, load_checkpoint
from functions.wb_eval import wb_eval, wb_train, wb_log_plot
from functions.xai import explain_dataset, evaluate_explainations
from utils.utils import enable_reproducibility
from functions.loss import load_loss_fun
import matplotlib.pyplot as plt
import numpy as np

def wb_test(seed, loss_name, lr, epoch, reg_rate):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=lr)
  if loss_name == "CrossEntropy":
    loss = load_loss_fun(loss_name)
  else:
    loss = load_loss_fun("RRR", reg_rate = reg_rate, normalize=True)
  
  train_set, val_set, test_set = load_data(
    "Waterbirds", reload=False, balance=True, seed=seed)
  data = [train_set, val_set, test_set]
  params = {"batch_size":64}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  log, dyn = wb_train(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss, 
    n_epochs=epoch, 
    eval_loader=val_loader, 
    patience=3,
    device=device
  )
  print(log)

  eval_loss = load_loss_fun("CrossEntropy")
  loss, acc, wga, gacc = wb_eval(model, test_loader, eval_loss, device)
  print("="*20,f"Test set Loss:{loss:.2f} | Acc:{acc:.2f}.","="*20)
  print(wga)
  print(gacc)
  wb_log_plot(log, f"{seed}_{loss_name}")


if __name__ == "__main__":
  ce = "CrossEntropy"
  rrr = "RRR"
  # First test confusion
  wb_test(123, ce, 1e-2, 10, 1e2) #87% test

  # Try different RRR configuration to find the best one
  wb_test(123, rrr, 1e-2, 100, 1) #89
  wb_test(123, rrr, 1e-2, 100, 1e1) #90
  wb_test(123, rrr, 1e-2, 100, 1e2) #86
  wb_test(123, rrr, 1e-2, 100, 1e3)
  wb_test(123, rrr, 1e-2, 100, 1e4)

  wb_test(123, rrr, 1e-3, 100, 1e2)
  wb_test(123, rrr, 1e-3, 100, 1e3)
  wb_test(123, rrr, 1e-3, 100, 1e4)

  wb_test(123, rrr, 1e-4, 100, 1e2)
  wb_test(123, rrr, 1e-4, 100, 1e3)
  wb_test(123, rrr, 1e-4, 100, 1e4)

  wb_test(123, rrr, 1e-1, 100, 1e2)
  wb_test(123, rrr, 1e-1, 100, 1e3)
  wb_test(123, rrr, 1e-1, 100, 1e4)