from mask_generator.evaluate import test_mnist, test_wb
from experiments.utils import enable_reproducibility
import torch

SEED = 123
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

if __name__ == "__main__":
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(SEED)

  #test_mnist(MODEL_ID, SEED, device, dataset="DecoyMNIST")
  test_wb(MODEL_ID, SEED, device)