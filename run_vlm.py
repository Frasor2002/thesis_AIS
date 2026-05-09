from mask_generator.evaluate import test_mnist, test_wb
from experiments.utils import enable_reproducibility
import torch

SEED = 123

if __name__ == "__main__":
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(SEED)

  test_wb(SEED, device)