from mask_generator.evaluate import test_mnist, test_wb, test_chc
from experiments.utils import enable_reproducibility
import torch

SEED = 123

# 8B params models:
#google/gemma-4-E4B-it
#Qwen/Qwen3-VL-8B-Instruct

#Bigger models:
#google/gemma-4-31B-it
#google/gemma-4-26B-A4B-it
#Qwen/Qwen3.6-27B
#Qwen/Qwen3.5-9B
#MODEL_ID = "gemma-4-31b-it" #API
#gemini-2.5-flash #API

#MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
#MODEL_ID = "gemini-2.5-flash"
#API = True
API = False
MODEL_ID = "google/gemma-4-E4B-it"

if __name__ == "__main__":
  use_cuda = torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'
  enable_reproducibility(SEED)

  test_mnist(MODEL_ID, SEED, device, dataset="DecoyMNIST", use_api=API)
  #test_chc(MODEL_ID,SEED, device, API)
  #test_wb(MODEL_ID, SEED, device, API)