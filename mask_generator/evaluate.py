from dataset.dataset import load_data
from mask_generator.vlm import load_VLM
from mask_generator.saliency import load_wb_saliency
import time


def test_mnist():
  # load data
  # load saliency maps

  # instantiate vlm
  # forward pass in vlm

  # viz and eval bb

  pass


def test_wb(seed):
  train_set, val_set, test_set = load_data(
    "Waterbirds", reload=False, balance=True, seed=seed
  )

  vlm = load_VLM("Qwen/Qwen3-VL-2B-Instruct")
  for i in range(3):
    id, img, y, mask = train_set[i]
    output = vlm.detect_confounders(img, label=y)
    print(f"Class {y}")
    print(f"Confounded {sum(mask) > 0}")
    print(output)

