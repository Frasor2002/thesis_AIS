from huggingface_hub import login, whoami
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import re
from PIL import Image

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(CURR_DIR, "log")

def convert_to_heatmap(sal_tensor: torch.Tensor, outlier_perc: float = 5.0) -> Image.Image:
  """
  Converts a raw attribution tensor into a Captum-style heatmap PIL Image.
  """
  sal_2d = torch.sum(torch.abs(sal_tensor), dim=0).detach().cpu().numpy()
  
  if outlier_perc > 0:
    vmax = np.percentile(sal_2d, 100 - outlier_perc)
    vmin = sal_2d.min()
    sal_2d = np.clip(sal_2d, vmin, vmax)
      
  vmax = sal_2d.max()
  vmin = sal_2d.min()
  if vmax > vmin:
    sal_2d = (sal_2d - vmin) / (vmax - vmin)
  else:
    sal_2d = np.zeros_like(sal_2d)
      
  cmap = plt.get_cmap('jet')
  heatmap_rgba = cmap(sal_2d)
  
  heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
  
  return Image.fromarray(heatmap_rgb)



def parse_bboxes(output_text: str):
  try:
    match = re.search(r"\[.*\]", output_text, re.DOTALL)
    if match is None: return []

    json_str = match.group(0)
    data = json.loads(json_str)

    bboxes = []
    for item in data:
      if "bbox" in item and len(item["bbox"]) == 4:
        bboxes.append(item["bbox"])

    return bboxes
  except Exception as e:
    print("Parsing failed:", e)
    return []

def bboxes_to_mask(bboxes, image_shape, normalize=True):
  H, W = image_shape
  mask = torch.zeros((H, W), dtype=torch.uint8)

  for xmin, ymin, xmax, ymax in bboxes:
    if normalize:
      xmin = int((xmin / 1000.0) * W)
      ymin = int((ymin / 1000.0) * H)
      xmax = int((xmax / 1000.0) * W)
      ymax = int((ymax / 1000.0) * H)

    # Clamp to make them stay inside bounds
    xmin = max(0, min(xmin, W))
    xmax = max(0, min(xmax, W))
    ymin = max(0, min(ymin, H))
    ymax = max(0, min(ymax, H))

    mask[ymin:ymax, xmin:xmax] = 1

  return mask

def log_vlm_output(dataset_name, sample_id, output_text):
  os.makedirs(LOG_PATH, exist_ok=True)
  log_filepath = os.path.join(LOG_PATH, f"{dataset_name}_vlm_out.txt") 
  with open(log_filepath, "a", encoding="utf-8") as f:
    f.write(f"=== Sample ID: {sample_id} ===\n")
    f.write(f"{output_text}\n")
    f.write("-" * 50 + "\n\n")

def log_metrics_output(dataset_name, sample_id, metrics_text):
  os.makedirs(LOG_PATH, exist_ok=True)
  log_filepath = os.path.join(LOG_PATH, f"{dataset_name}_metrics.txt") 
  with open(log_filepath, "a", encoding="utf-8") as f:
    f.write(f"=== Sample ID: {sample_id} ===\n")
    f.write(f"{metrics_text}\n")
    f.write("-" * 50 + "\n\n")

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

def evaluate_weighted_masks(y_true: torch.Tensor, y_pred: torch.Tensor, saliency: torch.Tensor) -> dict:
  y_true_bool = y_true.bool()
  y_pred_bool = y_pred.bool()

  sal_abs = torch.abs(saliency)
  
  if sal_abs.ndim == 3:
    sal_abs = torch.sum(sal_abs, dim=0)

  weighted_TP = (y_true_bool & y_pred_bool).float() * sal_abs
  weighted_FP = (~y_true_bool & y_pred_bool).float() * sal_abs
  weighted_FN = (y_true_bool & ~y_pred_bool).float() * sal_abs

  sum_TP = weighted_TP.sum().item()
  sum_FP = weighted_FP.sum().item()
  sum_FN = weighted_FN.sum().item()

  w_iou = sum_TP / (sum_TP + sum_FP + sum_FN) if (sum_TP + sum_FP + sum_FN) != 0 else 0.0
  w_precision = sum_TP / (sum_TP + sum_FP) if (sum_TP + sum_FP) != 0 else 0.0
  w_recall = sum_TP / (sum_TP + sum_FN) if (sum_TP + sum_FN) != 0 else 0.0

  return {
    "Weighted_IoU": w_iou,
    "Weighted_Precision": w_precision,
    "Weighted_Recall": w_recall
  }

def evaluate_plausibility(right_mask: torch.Tensor, wrong_mask: torch.Tensor) -> dict:
  right = right_mask.bool()
  wrong = wrong_mask.bool()

  intersection = (right & wrong).sum().item()
  
  union = (right | wrong).sum().item()

  overlap_ratio = intersection / union if union != 0 else 0.0

  plausibility = 1.0 - overlap_ratio

  return {
    "Plausibility_Score": plausibility,
    "Contradicting_Pixels": intersection,
    "Overlap_Ratio": overlap_ratio
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