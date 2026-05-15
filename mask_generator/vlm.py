from dotenv import load_dotenv
from mask_generator.utils import login_to_hub
import os
from mask_generator.vlm_models.qwen3 import load_qwen3_vl_instruct
from mask_generator.vlm_models.hf_vlm_loader import load_hf_vlm
from mask_generator.vlm_models.google_api import load_google_vlm


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(CURR_DIR, "prompt", "prompt.yaml")
QUAL_PROMPT_PATH = os.path.join(CURR_DIR, "prompt", "qualitative_prompt.yaml")

def load_VLM(model_id, use_api=True, **kwargs):
  load_dotenv()
  if not use_api: login_to_hub()

  prompt_path = QUAL_PROMPT_PATH

  if use_api:
    return load_google_vlm(model_id, prompt_path)

  # General hf loader should always work
  model = load_hf_vlm(model_id, prompt_path)

  #model = load_qwen3_vl_instruct(model_id, PROMPT_PATH)

  return model