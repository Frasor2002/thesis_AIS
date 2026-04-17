import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from model.model import load_model
from dataset.dataset import load_data, create_dataloaders
from functions.optimizer import load_optimizer
from functions.loss import load_loss_fun
from functions.functions import train_model, eval_model, save_checkpoint, load_checkpoint
from functions.xil import compute_simplicity
from utils.utils import enable_reproducibility
from experiments.utils import compute_correlations
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def visualize_pca_clusters(reduced_class: np.ndarray, cluster_labels: np.ndarray, 
                           class_confounded: np.ndarray, class_id: int):
    """
    Visualizes the PCA-reduced features, colored by predicted cluster 
    and shaped by true confounder presence.
    """
    plt.figure(figsize=(8, 6))

    # Combinations of (Cluster 0/1) and (Confounded True/False)
    # Cluster 0 - Unconfounded
    mask_c0_u = (cluster_labels == 0) & (class_confounded == 0)
    plt.scatter(reduced_class[mask_c0_u, 0], reduced_class[mask_c0_u, 1], 
                c='blue', marker='o', label='Cluster 0 (Unconfounded True)', alpha=0.6)
    
    # Cluster 0 - Confounded
    mask_c0_c = (cluster_labels == 0) & (class_confounded == 1)
    plt.scatter(reduced_class[mask_c0_c, 0], reduced_class[mask_c0_c, 1], 
                c='blue', marker='x', label='Cluster 0 (Confounded True)', alpha=0.8)

    # Cluster 1 - Unconfounded
    mask_c1_u = (cluster_labels == 1) & (class_confounded == 0)
    plt.scatter(reduced_class[mask_c1_u, 0], reduced_class[mask_c1_u, 1], 
                c='red', marker='o', label='Cluster 1 (Unconfounded True)', alpha=0.6)
    
    # Cluster 1 - Confounded
    mask_c1_c = (cluster_labels == 1) & (class_confounded == 1)
    plt.scatter(reduced_class[mask_c1_c, 0], reduced_class[mask_c1_c, 1], 
                c='red', marker='x', label='Cluster 1 (Confounded True)', alpha=0.8)

    plt.title(f'Latent Space Clustering for Class {class_id}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    


def extract_model_outputs(
  model: torch.nn.Module, 
  train_loader: DataLoader, 
  return_features: bool = False, 
  seed: int = 123, 
  device: str = "cpu") -> tuple:
  """Split the sample in confounded and unconfounded based on the model outputs.
  Args:
    model (Module): model to extract output from.
    train_loader (DataLoader): dataloader to extract outputs from.
    return_features (bool): if true return penultimate layer output and not logits.
    seed (int): seed for kmeans clustering.
    device (str): device where outputs are computed.
  Returns:
    tuple: separation list, confounder list and labels for correlation.
  """
  model.eval()
  outputs_list = []
  labels_list = []
  is_confounded_list = []

  with torch.no_grad():
    for _, imgs, targets, masks in train_loader:
      imgs = imgs.to(device)
      outputs = model(imgs, return_features)

      outputs_list.append(outputs.cpu().numpy())
      labels_list.append(targets.numpy())

      mask_sum = masks.view(masks.size(0), -1).sum(dim=1)
      is_confounded = (mask_sum > 0).numpy()
      is_confounded_list.append(is_confounded)
  
  outputs_all = np.concatenate(outputs_list, axis=0)
  labels_all = np.concatenate(labels_list, axis=0)
  is_confounded_all = np.concatenate(is_confounded_list, axis=0)


  # For each class, do PCA and get the two clusters
  is_in_cluster = np.zeros_like(labels_all)
  unique_classes = np.unique(labels_all)

  for c in unique_classes:
    class_mask = (labels_all == c)
    class_outputs = outputs_all[class_mask]
    class_confounded = is_confounded_all[class_mask]
        
    pca_class = PCA(n_components=2)
    reduced_class = pca_class.fit_transform(class_outputs)
    
    # Find the two clusters with GM
    gmm = GaussianMixture(n_components=2, random_state=seed)
    cluster_labels = gmm.fit_predict(reduced_class)
    #kmeans = KMeans(n_clusters=2, random_state=seed)
    #cluster_labels = kmeans.fit_predict(reduced_class)

    # Make sure that 1 are confounded and 0 unconfounded by checking the majority
    mean_confounded_c0 = class_confounded[cluster_labels == 0].mean() if sum(cluster_labels == 0) > 0 else 0
    mean_confounded_c1 = class_confounded[cluster_labels == 1].mean() if sum(cluster_labels == 1) > 0 else 0
        
    if mean_confounded_c0 > mean_confounded_c1:
      cluster_labels = np.subtract(1, cluster_labels)
    
    is_in_cluster[class_mask] = cluster_labels

    #visualize_pca_clusters(reduced_class, cluster_labels, class_confounded, c)
  
  return is_in_cluster, is_confounded_all, labels_all


def exp_model_outputs(
    seed: int = 123, 
    model_name :str = "LeNet",
    model_variant: str = "modern",
    dataset: str = "DecoyMNIST",
    variation: int = 0
    ) -> dict:
  """Run experiment to see how model output correlate with confounder presence.
  Args:
    seed (int): seed for the experiment.
    dataset (str): dataset to use for the experiment.
    metric (str): metric to use for simplicity.
  Returns:
    dict: correlation between confounder presence and model output.
  """
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'

  enable_reproducibility(seed)
  model = load_model(model_name, device=device)
  optim = load_optimizer("SGD", model.parameters(), lr=1e-2, weight_decay=0)
  loss = load_loss_fun("CrossEntropy")

  train_set, val_set, test_set = load_data(
    dataset, 
    seed=seed, 
    reload=True,
    bias_ratio=[0.99]*10,
    variation=variation
  )
  
  data = [train_set, val_set, test_set]
  params = {"batch_size":32}
  m_params = [params]*3
  train_loader, val_loader, test_loader = create_dataloaders(data, m_params)

  # After a few epochs of training the samples should be separable

  _, _ = train_model(
    model, 
    train_loader, 
    optim, 
    loss, 
    n_epochs=10, 
    eval_loader=val_loader, 
    device=device
  )

  separation_list, is_confounded, labels = extract_model_outputs(
    model, 
    train_loader, 
    return_features=True, 
    seed=seed,
    device=device
  )

  # Compute total and not total correlation
  result = compute_correlations(separation_list, is_confounded, labels)

  return result
