import torch
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.chc_eval import celeba_train, celeba_eval, celeba_log_plot
from functions.xai import explain_dataset, evaluate_explainations, visualize_k_expl
from utils.utils import enable_reproducibility
from functions.loss import load_loss_fun
import matplotlib.pyplot as plt

def chc_test(seed, loss_name, lr, epoch, reg_rate):
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(seed)

  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=lr)
  if loss_name == "CrossEntropy":
    loss = load_loss_fun(loss_name)
  else:
    loss = load_loss_fun("RRR", reg_rate = reg_rate, normalize=True)
  
  train_set, val_set, test_set = load_data("CelebAHC", reload=False)
  data = [train_set, val_set, test_set]
  params = {"batch_size":32}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  log, dyn = celeba_train(
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
  loss, acc, wga, gacc = celeba_eval(model, test_loader, eval_loss, device)
  print("="*20,f"Test set Loss:{loss:.2f} | Acc:{acc:.2f}.","="*20)
  print(wga)
  print(gacc)
  celeba_log_plot(log, f"{seed}_{loss_name}_{lr}_{reg_rate}")

  all_attr, all_imgs = explain_dataset(train_loader, model, device)
  exp_penalty, class_penalty = evaluate_explainations(all_attr, train_set.masks, train_set.y)
  print(exp_penalty)
  print(class_penalty)

  visualize_k_expl(all_attr, all_imgs, target_label=0)
  plt.savefig("Test1.pdf")
  visualize_k_expl(all_attr, all_imgs, target_label=1)
  plt.savefig("Test2.pdf")


if __name__ == "__main__":
  ce = "CrossEntropy"
  rrr = "RRR"
  chc_test(123, ce, 1e-1, 100, 1)
  chc_test(123, ce, 1e-2, 100, 1)


  # Try different RRR configuration to find the best one
  chc_test(123, rrr, 1e-2, 100, 1) #89
  chc_test(123, rrr, 1e-2, 100, 1e1) #90
  chc_test(123, rrr, 1e-2, 100, 1e2) #86
  chc_test(123, rrr, 1e-2, 100, 1e3)
  chc_test(123, rrr, 1e-2, 100, 1e4)

  chc_test(123, rrr, 1e-3, 100, 1e2)
  chc_test(123, rrr, 1e-3, 100, 1e3)
  chc_test(123, rrr, 1e-3, 100, 1e4)

  chc_test(123, rrr, 1e-4, 100, 1e2)
  chc_test(123, rrr, 1e-4, 100, 1e3)
  chc_test(123, rrr, 1e-4, 100, 1e4)

  chc_test(123, rrr, 1e-1, 100, 1e2)
  chc_test(123, rrr, 1e-1, 100, 1e3)
  chc_test(123, rrr, 1e-1, 100, 1e4)
  