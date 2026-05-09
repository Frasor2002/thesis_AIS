from dataset.dataset import load_data
from mask_generator.vlm import load_VLM
from mask_generator.saliency import load_mnist_saliency, load_wb_saliency
import time


def test_mnist(seed, device, dataset):
  train_set, _, _ = load_data(
    dataset, 
    seed=seed, 
    reload=True,
    bias_ratio=[0.99]*10,
    variation=2,
    train_patch=False
  )

  saliency_dict = load_mnist_saliency(seed, device, dataset)

  vlm = load_VLM("Qwen/Qwen3-VL-2B-Instruct")
  for i in range(3):
    id, img, y, mask = train_set[i]
    sal = saliency_dict[id]

    start_time = time.time()

    output = vlm.detect_confounders(img, saliency=sal, label=y)

    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Sample: {id}, Class {y}")
    print(f"Confounded {sum(mask) > 0}")
    print(f"Time for 1 sample {inference_time}")
    print(output)


def test_wb(seed, device):
  train_set, _, _ = load_data(
    "Waterbirds", reload=False, balance=True, seed=seed
  )

  saliency_dict = load_wb_saliency(seed, device)

  vlm = load_VLM("Qwen/Qwen3-VL-2B-Instruct")
  for i in range(3):
    id, img, y, mask = train_set[i]
    sal = saliency_dict[id]

    start_time = time.time()

    output = vlm.detect_confounders(img, saliency=sal, label=y)

    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Sample: {id}, Class {y}")
    print(f"Confounded {sum(mask) > 0}")
    print(f"Time for 1 sample {inference_time}")
    print(output)

