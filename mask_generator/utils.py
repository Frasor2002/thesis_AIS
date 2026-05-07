from huggingface_hub import login, whoami
import os
import torch

#TODO from bb to mask

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