import torch
import torch.nn as nn
import numpy as np
from captum.attr import Saliency, InputXGradient, IntegratedGradients, Attribution
from captum.attr import visualization as viz
from typing import Optional, Any
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.functional.classification import binary_auroc


def visualize_k_expl(all_attr: torch.Tensor, all_imgs: torch.Tensor, dataset: Any, target_label:int, k: int=3):
  """Visualizer helper to plot k attributions given a label.
  Args:
    all_attr (Tensor): all attributions.
    all_imgs (Tensor): all imgs to plot.
    dataset (Any): dataset to plot.
    target_label (int): label of which to plot explainations.
    k (int): number of explainations to show.
  """

  class_indices = np.where(dataset.y == target_label)[0]
  if len(class_indices) < k:
    print(f"Not enough samples for class {target_label}")
    selected_indices = class_indices
  else:
    selected_indices = np.random.permutation(class_indices)[:k]
  fig, axes = plt.subplots(1, k, figsize=(20, 5))
  for plot_i, data_idx in enumerate(selected_indices):
    visualize_explanation(
      attr=all_attr[data_idx], 
      img=all_imgs[data_idx],
      title=f"Class: {target_label}",
      plt_fig_axis=(fig, axes[plot_i]), 
      use_pyplot=False
    )
  plt.tight_layout()


def get_method(name: str, model: nn.Module) -> Attribution:
  """Get attribution method.
  Args:
    name (str): method name.
    model (nn.Module): model to explain.
  Returns:
    Attribution: attribution method.
  """
  attr_methods = {
    'input gradient': Saliency,
    'input X gradient': InputXGradient,
    'integrated gradient': IntegratedGradients,
  }

  if name not in attr_methods.keys():
    raise ValueError("Wrong explanation method name.")
  
  method_class = attr_methods[name]
  method = method_class(model)
  
  return method


def compute_explanation(method_name: str, model: nn.Module, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
  """Compute the explaination given a method name.
  Args:
    method_name (str): explaination method name.
    model (Module): model to explain.
    inputs (Tensor): batch of inputs to explain.
    targets (Optional[Tensor]): optional labels, if not present use model predictions.
  Returns:
    Tensor: computed explainations.
  """
  # Get model current device
  model.eval()
  device = next(model.parameters()).device

  # Expect an input of shape Batch_size X Channel X Height X Width
  inputs = inputs.to(device)
  explainer = get_method(method_name, model)

  # If no target is provided get predictions for inputs and use those
  if targets is None:
    logits = model(inputs)
    targets = logits.argmax(dim=1)

  attributions = explainer.attribute(inputs, targets)
  return attributions


def visualize_explanation(attr: torch.Tensor, img: torch.Tensor, **kwargs: Any) -> None:
  """Given a attributions and images, visualize the explaination.
  Args:
    attr (Tensor): either a single or a batch of attributions.
    img (Tensor): either a single or a batch of imgs.
  """

  def format_for_viz(tensor: torch.Tensor) -> np.ndarray:
    """Format a tensor for visualization."""
    if tensor.dim() == 4:
      tensor = tensor.squeeze(0) # Remove batch dimension
    # Permute to dimensions (H, W, C)
    return tensor.permute(1, 2, 0).detach().to("cpu").numpy()
  
  # Convert to correct format
  attr_np = format_for_viz(attr)
  img_np = format_for_viz(img)

  #print(f"Visualizing | Image Shape: {img_np.shape} | Attr Shape: {attr_np.shape}")

  # Call Captum Visualization
  viz.visualize_image_attr(
    attr_np,
    original_image=img_np,
    method="heat_map",
    show_colorbar=True,
    sign="absolute_value",
    outlier_perc=5,
    **kwargs
  )


def explain_dataset(loader: DataLoader, model: nn.Module, device: str="cpu") -> tuple:
  """Compute explaination for an entire dataset for visualization.
  Args:
    loader (DataLoader): dataloader with data to explain.
    model (Module): model to use for explanation.
    device (str): device where to compute explainations.
  """
  model.eval()
  attr_lists = []
  imgs_lists = []
  masks_lists = []

  loop = tqdm(loader, desc="Explaining", leave=False)
  for indices, imgs, targets, masks in loop:
    imgs = imgs.to(device)
    imgs.requires_grad_(True)
    attrs = compute_explanation("input gradient", model, imgs) 
    imgs.requires_grad_(False)

    # Save attrs
    attr_lists.append(attrs.detach().cpu())
    all_attr = torch.cat(attr_lists, dim=0)
    # Save imgs
    imgs_lists.append(imgs.detach().cpu())
    all_images = torch.cat(imgs_lists, dim=0)


  return all_attr, all_images


# TODO improve
def evaluate_explainations(pred_expl: torch.Tensor, gt_expl: Any, targets: Any) -> tuple:
  """Evaluate model explaination by computing a penalty.
  Args:
    pred_expl (Tensor): model attributions.
    gt_expl (Tensor): masks that contain confounder location.
  Returns:
    Any: penalty score.
  """
  pred = pred_expl.detach().cpu()
  gt = torch.as_tensor(gt_expl).detach().cpu()
  y = np.array(targets)
  # Compute multichannel attributions
  if pred.dim() == 4 and pred.shape[1] > 1:
    pred = torch.sum(torch.abs(pred), dim=1, keepdim=True)
  # Ensure mask has shape [B, 1, H, W]
  if gt.dim() == 3:
    gt = gt.unsqueeze(1)

  pred = pred ** 2
  batch_size = pred.shape[0]
  pred_flat = pred.reshape(batch_size, -1)
  gt_flat = gt.reshape(batch_size, -1)

  is_confounded = (torch.sum(gt_flat, dim=1) > 0)
  pred_conf = pred_flat[is_confounded]
  gt_conf = gt_flat[is_confounded]
  y_conf = y[is_confounded.numpy()]
    
  attr_on_conf = torch.sum(pred_conf * gt_conf, dim=1)
  total_attr = torch.sum(pred_conf, dim=1)
  attribution_percentage = torch.where(
    total_attr > 0, 
    attr_on_conf / total_attr, 
    torch.zeros_like(total_attr)
  )
  global_score = float(attribution_percentage.mean().item())

  num_total = batch_size
  num_confounded = is_confounded.sum().item()
  # = num_total - num_confounded
  
  #pct_confounded = (num_confounded / num_total) * 100
  #pct_unconfounded = (num_unconfounded / num_total) * 100
  
  #print(f"Total samples in batch: {num_total}")
  #print(f"Unconfounded (Mask == 0): {num_unconfounded} samples ({pct_unconfounded:.2f}%)")
  #print(f"Confounded   (Mask >  0): {num_confounded} samples ({pct_confounded:.2f}%)")    

  class_scores = {}
  unique_classes = np.unique(y_conf)
  for cls in unique_classes:
    idx = np.where(y_conf == cls)[0] 
    if len(idx) > 0:
      cls_score = float(attribution_percentage[idx].mean().item())
      class_scores[int(cls)] = cls_score
  
            
  return float(global_score), class_scores

  
