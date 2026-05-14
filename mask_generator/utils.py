from huggingface_hub import login, whoami
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(CURR_DIR, "log")


def format_saliency_for_vlm(saliency: torch.Tensor) -> torch.Tensor:
  sal = torch.abs(saliency)
  if sal.dim() == 3 and sal.shape[0] > 1:
    sal = torch.sum(sal, dim=0)
  elif sal.dim() == 3:
    sal = sal.squeeze(0)

  sal_min, sal_max = sal.min(), sal.max()
  if sal_max - sal_min > 1e-8:
    sal_norm = (sal - sal_min) / (sal_max - sal_min)
  else:
    sal_norm = sal

  cmap = plt.get_cmap('jet')
  sal_colored = cmap(sal_norm.cpu().numpy()) 

  return torch.tensor(sal_colored[..., :3]).permute(2, 0, 1).float().to(saliency.device)


def save_visualization(image, saliency, pred_mask, gt_mask, save_path, sample_id="", class_label=""):
  # Ensure the directory exists
  os.makedirs(LOG_PATH, exist_ok=True)
    
  fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
  def format_for_plotting(tensor):
    if hasattr(tensor, 'detach'): tensor = tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray) and tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
      tensor = np.transpose(tensor, (1, 2, 0))
    if isinstance(tensor, np.ndarray):
      tensor = np.squeeze(tensor)
    # Correct color normalization
    tensor = np.clip(tensor, 0.0, 1.0)
    return tensor

  # Format all inputs
  img_viz = format_for_plotting(image)
  sal_viz = format_for_plotting(saliency)
  pred_viz = format_for_plotting(pred_mask)
  gt_viz = format_for_plotting(gt_mask)

  # Input image
  axes[0].imshow(img_viz, cmap='binary' if img_viz.ndim == 2 else None)
  axes[0].set_title("Original Image", fontsize=14)
  axes[0].axis('off')
    
  # Saliency map
  axes[1].imshow(sal_viz, cmap='jet' if sal_viz.ndim == 2 else None)
  axes[1].set_title("Saliency Map", fontsize=14)
  axes[1].axis('off')
    
  # Predicted Mask
  axes[2].imshow(pred_viz, cmap='binary')
  axes[2].set_title("Predicted Mask", fontsize=14)
  axes[2].axis('off')
    
  # Ground Truth Mask
  axes[3].imshow(gt_viz, cmap='binary')
  axes[3].set_title("Ground Truth Mask", fontsize=14)
  axes[3].axis('off')
    
  # Optional Main Title
  if sample_id or class_label:
    plt.suptitle(f"Sample: {sample_id} | Class: {class_label}", fontsize=16)
    
  plt.tight_layout()
  path = os.path.join(LOG_PATH, save_path)
  plt.savefig(path)
  plt.close(fig)


def evaluate_masks(y_true: torch.Tensor , y_pred: torch.Tensor) -> dict:
  y_true = y_true.bool()
  y_pred = y_pred.bool()

  TP = (y_true & y_pred).sum().item()
  FP = (~y_true & y_pred).sum().item()
  FN = (y_true & ~y_pred).sum().item()
  TN = (~y_true & ~y_pred).sum().item()

  iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0.0
  dice = (2.0 * TP) / (2.0 * TP + FP + FN) if (2.0 * TP + FP + FN) != 0 else 0.0
  precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
  recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
  accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0

  return {
    "IoU": iou,
    "Dice": dice,
    "Precision": precision,
    "Recall": recall,
    "Accuracy": accuracy
  }


def login_to_hub() -> None:
  """Login to hugging face hub to load models."""
  # Check if already logged in
  try:
    user = whoami()
    print(f"hf user '{user['name']}' logged in.")
    return
  except Exception:
    pass
  
  # Login
  env_token = os.getenv("HF_TOKEN")
  if env_token:
    print("Logging in with HF_TOKEN...")
    login(token=env_token)
  else:
    print("WARNING: No authentication found. Set 'HF_TOKEN'.")