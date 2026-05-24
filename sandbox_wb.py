import torch
import numpy as np
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders, visualize_k_samples
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint, load_checkpoint
from functions.wb_eval import wb_eval, wb_train, wb_log_plot
from functions.xai import explain_dataset, evaluate_explainations, evaluate_confounder_dependence
from utils.utils import enable_reproducibility
from functions.loss import load_loss_fun
from experiments.utils import create_common_checkpoint
import matplotlib.pyplot as plt
import numpy as np

FREEZE = True
ONLY_CONF = False

def wb_test(seed,device, loss_name, lr, epoch, reg_rate):
  

  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True,freeze=FREEZE, device=device)
  load_checkpoint("reset_model_sandbox_wb", model, device)
  
  optim = load_optimizer("SGD", model.parameters(), lr=lr)
  if loss_name == "CrossEntropy":
    loss = load_loss_fun(loss_name)
  else:
    loss = load_loss_fun("RRR", reg_rate = reg_rate, normalize=True)
  
  train_set, val_set, test_set = load_data(
    "Waterbirds", reload=False, balance=True, seed=seed, only_conf=ONLY_CONF)
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
    #patience=3,
    device=device
  )
  print(log)

  eval_loss = load_loss_fun("CrossEntropy")
  loss, acc, wga, gacc = wb_eval(model, test_loader, eval_loss, device)
  print("="*20,f"Test set Loss:{loss:.2f} | Acc:{acc:.2f}.","="*20)
  print(wga)
  print(gacc)
  wb_log_plot(log, f"{seed}_{loss_name}_{lr}_{reg_rate}_{epoch}")

  all_attr, _ = explain_dataset(train_loader, model, device)
  exp_penalty, class_penalty = evaluate_explainations(all_attr, train_set.masks, train_set.y)
  print(exp_penalty)
  print(class_penalty)

  cd = evaluate_confounder_dependence(train_loader, model, device)
  print(cd)


SEED = 123

if __name__ == "__main__":
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(SEED)
  create_common_checkpoint(SEED, "ResNet", diff="_sandbox_wb", model_name="resnet50", n_classes=2, pretrained=True,freeze=FREEZE)

  ce = "CrossEntropy"
  rrr = "RRR"
  #wb_test(123, ce, 1e-1, 100, 1e2)
  #wb_test(123, rrr, 1e-1, 100, 1e2)

  # Different lr
  #wb_test(SEED,device, ce, 1e-1, 10, 1)
  wb_test(SEED,device, ce, 1e-2, 10, 1)
  wb_test(SEED,device, rrr, 1e-1, 10, 1e2)
