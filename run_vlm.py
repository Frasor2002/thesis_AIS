from mask_generator.evaluate import test_all_datasets
from mask_generator.saliency import save_all_data
from experiments.utils import enable_reproducibility
import torch

SEED = 123

AVAILABLE_MODELS = {
  "qwen_vl": "Qwen/Qwen3-VL-8B-Instruct",
  "gemma_31b": "google/gemma-4-31B-it",
  "gemma_26b": "google/gemma-4-26B-A4B-it",
  "qwen_27b": "Qwen/Qwen3.6-27B",
  "qwen_9b": "Qwen/Qwen3.5-9B",
  "gemma_e4b": "google/gemma-4-E4B-it",
}

#MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
#MODEL_ID = "gemini-2.5-flash"
API = False
MODEL_ID = "google/gemma-4-E2B-it"

if __name__ == "__main__":
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(SEED)

  ## 1. Generate and save the saliency samples
  save_all_data(SEED, device)

  # 2. Run the consolidated evaluation for all datasets
  #test_all_datasets(MODEL_ID, use_api=API)