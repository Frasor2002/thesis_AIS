import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from typing import Dict, Any, Union
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from typing import Optional
from dotenv import load_dotenv
from mask_generator.utils import login_to_hub
import os
import yaml

# Qwen/Qwen3.6-27B
# Qwen/Qwen3-VL-2B-Instruct
# Qwen/Qwen3-VL-4B-Instruct
# Qwen/Qwen3-VL-8B-Instruct
# Qwen/Qwen3.5-4B
# google/gemma-3-4b-it

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(CURR_DIR, "prompt", "prompt.yaml")

# Currently only compatible with Qwen3
#TODO add compatibility to other models
class VLM:
  def __init__(self, model_id: str):
    self.model_id = model_id

    print(f"Loading {self.model_id}")

    self.processor = AutoProcessor.from_pretrained(
      self.model_id, 
      trust_remote_code=True
    )

    # if "Qwen3" in model_id:
    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
      model_id,
      dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
      device_map="auto"
    ).eval()
    
  
  def _load_prompt(self) -> dict:
    if not os.path.exists(PROMPT_PATH):
      raise FileNotFoundError(f"Prompt file not found at: {PROMPT_PATH}")
            
    with open(PROMPT_PATH, "r", encoding="utf-8") as file:
      prompt_data = yaml.safe_load(file)
            
    return prompt_data


  
  def detect_confounders(
    self,
    img: Tensor,
    label: str,
    saliency: Optional[Tensor] = None,
  ):
    prompt_dict = self._load_prompt()


    images_to_process = [img]
    if saliency is not None:
      images_to_process.append(saliency)
      prompt_template = prompt_dict["prompt"]
    else:
      prompt_template = prompt_dict["prompt_not_sal"]
    
    # Add the label to the prompt
    prompt_text = prompt_template.format(label=label)

    # Build the message content list
    content = []
    for image in images_to_process:
      content.append({"type": "image", "image": to_pil_image(image)})      
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]

    inputs = self.processor.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=True,
      return_dict=True,
      return_tensors="pt"
    )
    inputs = inputs.to(self.model.device)

    with torch.no_grad(): 
      generated_ids = self.model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
      out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = self.processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return output_text



def load_VLM(model_id):
  load_dotenv()
  login_to_hub()

  return VLM(model_id)