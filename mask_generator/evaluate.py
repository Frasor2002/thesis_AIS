from mask_generator.vlm import load_VLM
from mask_generator.saliency import load_saved_dataset
from mask_generator.utils import save_visualization, evaluate_masks, format_saliency_for_vlm, parse_bboxes, bboxes_to_mask, log_vlm_output
import time

def test_mnist(model_id, seed, device, dataset, use_api):
  # Load the VLM and pre-saved samples directly
  vlm = load_VLM(model_id, use_api)
  samples = load_saved_dataset(dataset)
  
  for i, s in enumerate(samples):
    item_id = s.get("id", i)
    img = s["img"]
    pred = s["pred"]
    lab = s["label"]
    sal = format_saliency_for_vlm(s["saliency"])
    mask = s.get("mask", None)

    start_time = time.time()
    output = vlm.detect_confounders(img, saliency=sal, label=lab, pred=pred)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Sample: {item_id}, Class: {lab}")
    if mask is not None:
      print(f"Confounded: {mask.sum() > 1}")
    print(f"Time for 1 sample: {inference_time}")

    print(output)
    log_vlm_output(dataset, i, output)

    # Handle shape whether img is a PIL Image or PyTorch Tensor
    #shape = img.size[::-1] if hasattr(img, 'size') else img.shape[-2:]
    #output = bboxes_to_mask(parse_bboxes(output), shape, normalize=True)
    
    #save_visualization(img, sal, output, mask, f"{dataset}_{i}.pdf", item_id, lab)
    # if mask is not None:
    #   print(evaluate_masks(mask, output))


def test_wb(model_id, seed, device, use_api):
  vlm = load_VLM(model_id, use_api)
  samples = load_saved_dataset("Waterbirds")

  for i, s in enumerate(samples):
    item_id = s.get("id", i)
    img = s["img"]
    pred = s["pred"]
    lab = s["label"]
    sal = format_saliency_for_vlm(s["saliency"])
    mask = s.get("mask", None)

    start_time = time.time()
    output = vlm.detect_confounders(img, saliency=sal, label=lab, pred=pred)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Sample: {item_id}, Class: {lab}")
    if mask is not None:
      print(f"Confounded: {mask.sum() > 1}")
    print(f"Time for 1 sample: {inference_time}")

    print(output)
    log_vlm_output("Waterbirds", i, output)

    #shape = img.size[::-1] if hasattr(img, 'size') else img.shape[-2:]
    #output = bboxes_to_mask(parse_bboxes(output), shape, normalize=True)

    #save_visualization(img, sal, output, mask, f"wb_{i}.pdf", item_id, lab)
    # if mask is not None:
    #   print(evaluate_masks(mask, output))


def test_chc(model_id, seed, device, use_api):
  vlm = load_VLM(model_id, use_api)
  samples = load_saved_dataset("CelebAHC")

  for i, s in enumerate(samples):
    item_id = s.get("id", i)
    img = s["img"]
    pred = s["pred"]
    lab = s["label"]
    sal = format_saliency_for_vlm(s["saliency"])
    mask = s.get("mask", None)

    start_time = time.time()
    output = vlm.detect_confounders(img, saliency=sal, label=lab, pred=pred)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Sample: {item_id}, Class: {lab}")
    if mask is not None:
      print(f"Confounded: {mask.sum() > 1}")
    print(f"Time for 1 sample: {inference_time}")

    print(output)
    log_vlm_output("CelebAHC", i, output)
    
    #shape = img.size[::-1] if hasattr(img, 'size') else img.shape[-2:]
    #output = bboxes_to_mask(parse_bboxes(output), shape, normalize=True)

    #save_visualization(img, sal, output, mask, f"celeba_{i}.pdf", item_id, lab)
    # if mask is not None:
    #   print(evaluate_masks(mask, output))