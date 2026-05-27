import os
import json
import torch
from torchvision.utils import save_image
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model
from functions.wb_eval import wb_train
from functions.chc_eval import celeba_train
from functions.xai import explain_dataset_with_predictions
from PIL import Image
from mask_generator.utils import convert_to_heatmap


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SALIENCY_PATH = os.path.join(CURR_DIR, "saliency")
DATA_PATH = os.path.join(CURR_DIR, "data")

def prepare_mnist_saliency(seed, device, dataset):
  model = load_model("ModernLeNet", device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2)
  loss = load_loss_fun("CrossEntropy")
  
  train_set, val_set, test_set = load_data(
    dataset, 
    seed=seed, 
    reload=True,
    bias_ratio=[0.99]*10,
    variation=2,
    train_patch=False
  )
  data = [train_set, val_set, test_set]
  params = {"batch_size":32}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  log, _ = train_model(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss, 
    n_epochs=10, 
    eval_loader=val_loader, 
    device=device
  )
  
  # Unpack predictions alongside attributions
  all_attr, _, all_preds = explain_dataset_with_predictions(train_loader, model, device)
  os.makedirs(SALIENCY_PATH, exist_ok=True)
  saliency_dict = {}

  # Save both attribution and prediction in the dictionary
  for i in range(len(train_set)):
    item_id = train_set[i][0]
    saliency_dict[item_id] = {
      "saliency": all_attr[i],
      "prediction": all_preds[i]
    }
  
  save_path = os.path.join(SALIENCY_PATH, f"{dataset}_sal.pth")
  torch.save(saliency_dict, save_path)
  print(f"Successfully saved {len(saliency_dict)} saliency maps to '{save_path}'")


def load_mnist_saliency(seed, device, dataset):
  load_path = os.path.join(SALIENCY_PATH, f"{dataset}_sal.pth")
  if not os.path.exists(load_path):
    prepare_mnist_saliency(seed, device, dataset)
    
  saliency_dict = torch.load(load_path, map_location=device, weights_only=False)
  print(f"Loaded {len(saliency_dict)} saliency maps from '{load_path}'")
    
  return saliency_dict


def prepare_wb_saliency(seed, device):
  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2)
  loss = load_loss_fun("CrossEntropy")
  train_set, val_set, test_set = load_data("Waterbirds", reload=False, balance=True, seed=seed)
  data = [train_set, val_set, test_set]
  params = {"batch_size":64}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)
  _, _ = wb_train(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss, 
    n_epochs=10,
    eval_loader=val_loader, 
    device=device
  )
  
  # Unpack predictions
  all_attr, _, all_preds = explain_dataset_with_predictions(train_loader, model, device)
  os.makedirs(SALIENCY_PATH, exist_ok=True)
  saliency_dict = {}

  # Save both attribution and prediction
  for i in range(len(train_set)):
    item_id = train_set[i][0]
    saliency_dict[item_id] = {
      "saliency": all_attr[i],
      "prediction": all_preds[i]
    }
  
  save_path = os.path.join(SALIENCY_PATH, "wb_sal.pth")
  torch.save(saliency_dict, save_path)
  print(f"Successfully saved {len(saliency_dict)} saliency maps to '{save_path}'")


def load_wb_saliency(seed, device="cuda"):
  load_path = os.path.join(SALIENCY_PATH, "wb_sal.pth")
  if not os.path.exists(load_path):
    prepare_wb_saliency(seed, device)
    
  saliency_dict = torch.load(load_path, map_location=device, weights_only=False)
  print(f"Loaded {len(saliency_dict)} saliency maps from '{load_path}'")
    
  return saliency_dict


def prepare_celeba_saliency(seed, device):
  model = load_model("ResNet", model_name="resnet50", n_classes=2, pretrained=True, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-3)
  loss = load_loss_fun("CrossEntropy")
  train_set, val_set, test_set = load_data("CelebAHC", reload=False)
  data = [train_set, val_set, test_set]
  params = {"batch_size": 64}
  m_params = [params] * 3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)
  
  _, _ = celeba_train(
    model=model, 
    train_loader=train_loader, 
    optimizer=optim, 
    loss_fun=loss, 
    n_epochs=50, 
    eval_loader=val_loader, 
    device=device
  )
  
  # Unpack predictions
  all_attr, _, all_preds = explain_dataset_with_predictions(train_loader, model, device)
  os.makedirs(SALIENCY_PATH, exist_ok=True)
  saliency_dict = {}

  # Save both attribution and prediction
  for i in range(len(train_set)):
    item_id = train_set[i][0]
    saliency_dict[item_id] = {
      "saliency": all_attr[i],
      "prediction": all_preds[i]
    }
  
  save_path = os.path.join(SALIENCY_PATH, "celeba_sal.pth")
  torch.save(saliency_dict, save_path)
  print(f"Successfully saved {len(saliency_dict)} saliency maps to '{save_path}'")


def load_celeba_saliency(seed, device="cuda"):
  load_path = os.path.join(SALIENCY_PATH, "celeba_sal.pth")
  if not os.path.exists(load_path):
    prepare_celeba_saliency(seed, device)
    
  saliency_dict = torch.load(load_path, map_location=device, weights_only=False)
  print(f"Loaded {len(saliency_dict)} saliency maps from '{load_path}'")
    
  return saliency_dict


def saliency_sampler(dataset, saliency_dict, n_classes, k):
  if k % 4 != 0:
    raise ValueError("k must be a multiple of 4 to sample equally between the 4 subgroups (confounded/unconfounded x correct/wrong).")

  quarter_k = k // 4
  
  # Structure: buckets[class_label][is_confounded][is_correct]
  buckets = {
    y: {
      True: {True: [], False: []}, 
      False: {True: [], False: []}
    } for y in range(n_classes)
  }
  
  # Group samples by class, confounded status, and prediction correctness
  for i in range(len(dataset)):
    item_id, img, label, mask = dataset[i]
    
    if item_id not in saliency_dict:
      continue
      
    dict_entry = saliency_dict[item_id]
    sal_map = dict_entry["saliency"]
    pred_val = dict_entry["prediction"]
    
    y = label.item() if isinstance(label, torch.Tensor) else label
    p = pred_val.item() if isinstance(pred_val, torch.Tensor) else pred_val
    
    is_confounded = True if mask.sum().item() > 1 else False
    is_correct = (y == p)
    
    # Calculate maximum saliency for sorting
    max_sal = sal_map.max().item() if isinstance(sal_map, torch.Tensor) else sal_map.max()
    
    buckets[y][is_confounded][is_correct].append({
      "idx": i,
      "max_sal": max_sal,
      "sal_map": sal_map,
      "pred_val": p,
      "is_correct": is_correct
    })
      
  sampled_data = []
  
  # Sample from each of the 4 buckets per class
  for y in range(n_classes):
    for is_confounded in [True, False]:
      for is_correct in [True, False]:
        group_items = buckets[y][is_confounded][is_correct]
        
        # Sort by max saliency (descending order)
        group_items.sort(key=lambda x: x["max_sal"], reverse=True)
        sampled_items = group_items[:quarter_k]
        
        if len(sampled_items) < quarter_k:
          print(f"Warning: Only found {len(sampled_items)} samples for class {y}, confounded={is_confounded}, correct={is_correct} (requested {quarter_k}).")
            
        for item in sampled_items:
          idx = item["idx"]
          item_id, img, _, mask = dataset[idx]
          
          corr_str = "right prediction" if is_correct else "wrong prediction"
          conf_str = "confounded" if is_confounded else "not confounded"
          category = f"{corr_str}, {conf_str}"

          sampled_data.append({
            "id": item_id,
            "img": img,
            "label": y,
            "mask": mask,
            "confounded": int(is_confounded),
            "saliency": item["sal_map"],
            "prediction": item["pred_val"],
            "category": category
          })
          
  return sampled_data

# Dicts to go from int to string for labels
fmnist_to_string = {0: "t-shirt/top",1: "trouser",2: "pullover",3: "dress",4: "coat",5: "sandal",6: "shirt",7: "sneaker",8: "bag",9: "ankle boot"}
wb_to_string = {1: "waterbird", 0: "landbird"}
celebahc_to_string = {1: "blond hair", 0: "not blond hair"}
lab_to_str = {
  "DecoyFMNIST": fmnist_to_string,
  "Waterbirds": wb_to_string,
  "CelebAHC": celebahc_to_string
}

def save_dataset(samples, data_name):
  dataset_dir = os.path.join(DATA_PATH, data_name)
  os.makedirs(dataset_dir, exist_ok=True)
  
  for k, s in enumerate(samples):
    sample_dir = os.path.join(dataset_dir, f"sample_{k}")
    os.makedirs(sample_dir, exist_ok=True)
    
    img_path = os.path.join(sample_dir, "img.png")
    sal_path = os.path.join(sample_dir, "sal.png")
    info_path = os.path.join(sample_dir, "info.json")
    mask_path = os.path.join(sample_dir, "mask.pt")
    
    # Save Image
    img_tensor = s["img"]
    save_image(img_tensor, img_path)
    
    # Save Saliency Map with normalization
    saliency = s["saliency"]
    heatmap_image = convert_to_heatmap(saliency, outlier_perc=5.0)
    heatmap_image.save(sal_path)

    # Save Ground Truth and Prediction
    label_val = int(s["label"])
    pred_val = int(s["prediction"])

    # Save Ground Truth maks
    torch.save(s["mask"], mask_path)

    if data_name == "DecoyFashionMNIST":
      label_str = fmnist_to_string[label_val]
      pred_str = fmnist_to_string[pred_val]
    elif data_name == "Waterbirds":
      label_str = wb_to_string[label_val]
      pred_str = wb_to_string[pred_val]
    elif data_name == "CelebAHC":
      label_str = celebahc_to_string[label_val]
      pred_str = celebahc_to_string[pred_val]
    else:
      label_str = str(label_val)
      pred_str = str(pred_val)

    # Include the category tag in info.json
    info_dict = {
      "ground_truth_label": label_str,
      "prediction": pred_str,
      "category": s["category"]
    }
    
    with open(info_path, "w") as f:
      json.dump(info_dict, f, indent=2)

  print(f"Successfully exported {len(samples)} samples to '{dataset_dir}'")


def save_all_data(seed, device): # do for all classes like in evaluate
  # 100 samples per dataset -> 400 samples

  # 1. Decoy MNIST
  print("\nProcessing DecoyMNIST...")
  saliency_dict_mnist = load_mnist_saliency(seed, device, "DecoyMNIST")
  train_set_mnist, _, _ = load_data("DecoyMNIST", seed=seed, reload=True, bias_ratio=[0.99]*10, variation=2, train_patch=False)
  samples_mnist = saliency_sampler(train_set_mnist, saliency_dict_mnist, n_classes=10, k=10)
  save_dataset(samples_mnist, "DecoyMNIST")

  # 2. Decoy Fashion MNIST
  print("\nProcessing DecoyFashionMNIST...")
  saliency_dict_fmnist = load_mnist_saliency(seed, device, "DecoyFashionMNIST")
  train_set_fmnist, _, _ = load_data("DecoyFashionMNIST", seed=seed, reload=True, bias_ratio=[0.99]*10, variation=2, train_patch=False)
  samples_fmnist = saliency_sampler(train_set_fmnist, saliency_dict_fmnist, n_classes=10, k=10)
  save_dataset(samples_fmnist, "DecoyFashionMNIST")
  
  # 3. Waterbirds
  print("\nProcessing Waterbirds...")
  saliency_dict_wb = load_wb_saliency(seed, device)
  train_set_wb, _, _ = load_data("Waterbirds", reload=False, balance=True, seed=seed)
  samples_wb = saliency_sampler(train_set_wb, saliency_dict_wb, n_classes=2, k=50)
  save_dataset(samples_wb, "Waterbirds")
  
  # 4. CelebAHC
  print("\nProcessing CelebAHC...")
  saliency_dict_celeba = load_celeba_saliency(seed, device)
  train_set_celeba, _, _ = load_data("CelebAHC", reload=False)
  samples_celeba = saliency_sampler(train_set_celeba, saliency_dict_celeba, n_classes=2, k=50)
  save_dataset(samples_celeba, "CelebAHC")


def load_saved_dataset(data_name):
  dataset_dir = os.path.join(DATA_PATH, data_name)
  
  if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
      
  samples = []
  
  sample_folders = [f for f in os.listdir(dataset_dir) if f.startswith("sample_")]
  sample_folders.sort(key=lambda x: int(x.split('_')[1]))
  
  for folder in sample_folders:
    sample_dir = os.path.join(dataset_dir, folder)
    
    img_path = os.path.join(sample_dir, "img.png")
    sal_path = os.path.join(sample_dir, "sal.png")
    info_path = os.path.join(sample_dir, "info.json")
    mask_path = os.path.join(sample_dir, "mask.pt")
    
    # Load Images as PIL (keep them as PIL for the HF pipeline)
    try:
      img_pil = Image.open(img_path).convert("RGB")
      sal_pil = Image.open(sal_path).convert("RGB")
      mask_tensor = torch.load(mask_path)
    except (FileNotFoundError, OSError) as e:
      print(f"Warning: Skipping {folder} due to missing or corrupted image files. ({e})")
      continue
      
    try:
      with open(info_path, "r") as f:
        info_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
      print(f"Warning: Skipping {folder} due to missing or invalid info.json. ({e})")
      continue
      
    samples.append({
      "img": img_pil,
      "saliency": sal_pil,
      "mask": mask_tensor,
      "label": info_dict["ground_truth_label"],
      "pred": info_dict["prediction"],
      "category": info_dict["category"]
    })
      
  print(f"Successfully loaded {len(samples)} samples from '{dataset_dir}'")
  return samples

