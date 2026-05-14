from dataset.dataset import load_data
from mask_generator.vlm import load_VLM
from mask_generator.saliency import load_mnist_saliency, load_wb_saliency, load_celeba_saliency, saliency_sampler
from mask_generator.utils import save_visualization, evaluate_masks, format_saliency_for_vlm
import time


def test_mnist(model_id, seed, device, dataset):
  fmnist_to_string = {
    0: "t-shirt/top",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle boot"
  }

  train_set, _, _ = load_data(
    dataset, 
    seed=seed, 
    reload=True,
    bias_ratio=[0.99]*10,
    variation=2,
    train_patch=False
  )

  saliency_dict = load_mnist_saliency(seed, device, dataset)
  vlm = load_VLM(model_id)

  # Use the sampler (e.g., k=4 samples per class)
  samples = saliency_sampler(train_set, saliency_dict, k=4, n_classes=10)
  
  for i, s in enumerate(samples):
    id = s["id"]
    img = s["img"]
    y = s["label"]
    sal = format_saliency_for_vlm(s["saliency"])
    mask = s["mask"] 

    if dataset == "DecoyMNIST": lab = str(y)
    else: lab = fmnist_to_string[y]

    start_time = time.time()
    output = vlm.detect_confounders(img, saliency=sal, label=lab)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Sample: {id}, Class {y}")
    print(f"Confounded {mask.sum() > 1}")
    print(f"Time for 1 sample {inference_time}")
    
    save_visualization(img, sal, output, mask, f"{dataset}_{i}.pdf", id, lab)
    print(evaluate_masks(mask, output))


def test_wb(model_id, seed, device):
  train_set, _, _ = load_data(
    "Waterbirds", reload=False, balance=True, seed=seed
  )

  lab_to_string = {
    1: "waterbird", 0: "landbird"
  }

  saliency_dict = load_wb_saliency(seed, device)
  vlm = load_VLM(model_id)
  
  # Use the sampler (e.g., k=10 samples per class) # k*2 total samples
  samples = saliency_sampler(train_set, saliency_dict, k=4, n_classes=2)

  for i, s in enumerate(samples):
    id = s["id"]
    img = s["img"]
    y = s["label"]
    sal = format_saliency_for_vlm(s["saliency"])
    mask = s["mask"]

    lab = lab_to_string[y]

    start_time = time.time()
    output = vlm.detect_confounders(img, saliency=sal, label=lab)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Sample: {id}, Class {y}")
    print(f"Confounded {mask.sum() > 1}")
    print(f"Time for 1 sample {inference_time}")
    save_visualization(img, sal, output, mask, f"wb_{i}.pdf", id, lab)
    print(evaluate_masks(mask, output))


def test_chc(model_id, seed, device):
  train_set, _, _ = load_data(
    "CelebAHC", reload=False
  )

  # CelebA Hair Color labels
  lab_to_string = {
    1: "blond hair", 0: "not blond hair"
  }

  saliency_dict = load_celeba_saliency(seed, device)
  vlm = load_VLM(model_id)

  # Use the sampler (e.g., k=4 samples per class)
  samples = saliency_sampler(train_set, saliency_dict, k=4, n_classes=2)

  for i, s in enumerate(samples):
    id = s["id"]
    img = s["img"]
    y = s["label"]
    sal = format_saliency_for_vlm(s["saliency"])
    mask = s["mask"]

    lab = lab_to_string[y]

    start_time = time.time()
    output = vlm.detect_confounders(img, saliency=sal, label=lab)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Sample: {id}, Class {y}")
    print(f"Confounded {mask.sum() > 1}")
    print(f"Time for 1 sample {inference_time}")
    
    save_visualization(img, sal, output, mask, f"celeba_{i}.pdf", id, lab)
    print(evaluate_masks(mask, output))