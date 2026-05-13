from cmd import PROMPT

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from typing import Dict, Any, Union
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from typing import Optional
from dotenv import load_dotenv
from mask_generator.utils import login_to_hub
import os
import yaml
import json
import re
from mask_generator.vlm_models.qwen3 import load_qwen3_vl_instruct
from mask_generator.vlm_models.hf_vlm_loader import load_hf_vlm

# Qwen/Qwen3.6-27B
# Qwen/Qwen3.5-4B
# google/gemma-3-4b-it

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(CURR_DIR, "prompt", "prompt.yaml")



def load_VLM(model_id, **kwargs):
  load_dotenv()
  login_to_hub()

  if "Qwen3" in model_id:
    model = load_qwen3_vl_instruct(model_id, PROMPT_PATH)
  else:
    model = load_hf_vlm(model_id, PROMPT_PATH)

  return model