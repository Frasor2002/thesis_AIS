from mask_generator.vlm import load_VLM
from mask_generator.saliency import load_saved_dataset
from mask_generator.utils import (
  save_visualization, 
  evaluate_masks, 
  parse_bboxes, 
  bboxes_to_mask, 
  log_vlm_output,
  log_metrics_output,
  evaluate_weighted_masks, 
  evaluate_plausibility
)
import time
import torch
import numpy as np
import json

def evaluate_dataset(vlm, model_name, dataset_name, prefix):
  print(f"\n========== Evaluating {model_name} on {dataset_name} ==========")
  samples = load_saved_dataset(dataset_name)
  num_samples = len(samples)
  
  if num_samples == 0:
    print(f"No samples found for {dataset_name}.")
    return

  # Accumulators for average calculations
  std_totals = {"IoU": 0.0, "Dice": 0.0, "Precision": 0.0, "Recall": 0.0, "Accuracy": 0.0}
  w_totals = {"Weighted_IoU": 0.0, "Weighted_Precision": 0.0, "Weighted_Recall": 0.0}
  plaus_totals = {"Plausibility_Score": 0.0, "Contradicting_Pixels": 0.0, "Overlap_Ratio": 0.0}

  for i, s in enumerate(samples):
    item_id = i # Id is number in the folder
    img = s["img"] # PIL image
    pred = s["pred"] # str prediction
    lab = s["label"] # str prediction
    sal = s["saliency"] # PIL image
    mask = s["mask"] # Tensor

    # Generate wrong reason mask in the image
    start_time = time.time()
    output_wr = vlm.detect_confounders(img, saliency=sal, label=lab, pred=pred, rr=False)
    wr_time = time.time() - start_time

    # Generate right reason mask in the image
    start_time = time.time()
    output_rr = vlm.detect_confounders(img, saliency=sal, label=lab, pred=pred, rr=True)
    rr_time = time.time() - start_time

    print(f"\nSample: {item_id}, Class: {lab} | Confounded GT: {mask.sum() > 1}")
    print(f"Time for WR: {wr_time:.2f}s | RR: {rr_time:.2f}s")

    # Convert outputs to masks
    shape = img.shape[-2:]
    pred_mask_wr = bboxes_to_mask(parse_bboxes(output_wr), shape, normalize=False)
    pred_mask_rr = bboxes_to_mask(parse_bboxes(output_rr), shape, normalize=False)
    
    # Standard Metrics 
    std_metrics = evaluate_masks(mask, pred_mask_wr)
    
    # Weighted Metrics
    sal_tensor = torch.tensor(np.array(sal.convert("L")), dtype=torch.float32) / 255.0
    w_metrics = evaluate_weighted_masks(mask, pred_mask_wr, sal_tensor)
    
    # Plausibility Metric
    plaus_metrics = evaluate_plausibility(pred_mask_rr, pred_mask_wr)

    # Accumulate totals for averaging later
    for k in std_totals: std_totals[k] += std_metrics[k]
    for k in w_totals: w_totals[k] += w_metrics[k]
    for k in plaus_totals: plaus_totals[k] += plaus_metrics[k]

    # Format pure text for the VLM log
    vlm_log_text = (
      f"--- WRONG REASONS (Confounders) ---\n{output_wr}\n\n"
      f"--- RIGHT REASONS (Valid Features) ---\n{output_rr}"
    )

    # Format JSON strings for the metrics log
    metrics_log_str = (
      f"Standard: {json.dumps(std_metrics)}\n"
      f"Weighted: {json.dumps(w_metrics)}\n"
      f"Plausibility: {json.dumps(plaus_metrics)}"
    )
    
    # Log to the two separate files
    log_vlm_output(model_name, dataset_name, i, vlm_log_text)
    log_metrics_output(model_name, dataset_name, i, metrics_log_str)
    
    save_visualization(img, sal, pred_mask_wr, mask, f"{model_name}_{prefix}_{i}.pdf", item_id, lab)


  print(f"\nComputing averages for {dataset_name} across {num_samples} samples...")
  
  std_avgs = {k: v / num_samples for k, v in std_totals.items()}
  w_avgs = {k: v / num_samples for k, v in w_totals.items()}
  plaus_avgs = {k: v / num_samples for k, v in plaus_totals.items()}

  print(f"Avg Standard Metrics: {json.dumps(std_avgs, indent=2)}")
  print(f"Avg Weighted Metrics: {json.dumps(w_avgs, indent=2)}")
  print(f"Avg Plausibility: {json.dumps(plaus_avgs, indent=2)}")

  avg_log_str = (
    f"Average Standard: {json.dumps(std_avgs, indent=2)}\n"
    f"Average Weighted: {json.dumps(w_avgs, indent=2)}\n"
    f"Average Plausibility: {json.dumps(plaus_avgs, indent=2)}\n"
  )
  
  # Log the final averages exclusively to the metrics file
  log_metrics_output(dataset_name, "FINAL AVERAGES", avg_log_str)


def test_all_datasets(model_id, use_api=False):
  """
  Master function to load the VLM once and test all required datasets.
  """
  print("Loading VLM into memory...")
  vlm = load_VLM(model_id, use_api=use_api)
  print("VLM Loaded successfully. Beginning bulk evaluation.\n")

  # Define the datasets and their file prefixes
  datasets_to_test = [
    ("DecoyMNIST", "mnist"),
    ("DecoyFashionMNIST", "fmnist"),
    ("Waterbirds", "wb"),
    ("CelebAHC", "chc")
  ]

  for dataset_name, prefix in datasets_to_test:
    try:
      evaluate_dataset(vlm, model_id, dataset_name, prefix)
    except FileNotFoundError:
      print(f"Skipping {dataset_name} - Dataset directory not found.")
    except Exception as e:
      print(f"An error occurred while evaluating {dataset_name}: {e}")

  print("\n========== All Evaluations Completed ==========")