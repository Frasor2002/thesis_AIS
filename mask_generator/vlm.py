import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
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

# Qwen/Qwen3.6-27B
# Qwen/Qwen3-VL-2B-Instruct
# Qwen/Qwen3-VL-4B-Instruct
# Qwen/Qwen3-VL-8B-Instruct
# Qwen/Qwen3.5-4B
# google/gemma-3-4b-it

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(CURR_DIR, "prompt", "prompt.yaml")



def parse_bboxes(output_text: str):
  try:
    match = re.search(r"\[.*\]", output_text, re.DOTALL)
    if match is None: return []

    json_str = match.group(0)
    data = json.loads(json_str)

    bboxes = []
    for item in data:
      if "bbox" in item and len(item["bbox"]) == 4:
        bboxes.append(item["bbox"])

    return bboxes
  except Exception as e:
    print("Parsing failed:", e)
    return []
  
def bboxes_to_mask(bboxes, image_shape):
  H, W = image_shape
  mask = torch.zeros((H, W), dtype=torch.uint8)

  for xmin, ymin, xmax, ymax in bboxes:
    # Move images from 0-1000 range to the actual image size
    xmin = int((xmin / 1000.0) * W)
    ymin = int((ymin / 1000.0) * H)
    xmax = int((xmax / 1000.0) * W)
    ymax = int((ymax / 1000.0) * H)

    # Clamp to make them stay inside bounds
    xmin = max(0, min(xmin, W))
    xmax = max(0, min(xmax, W))
    ymin = max(0, min(ymin, H))
    ymax = max(0, min(ymax, H))

    mask[ymin:ymax, xmin:xmax] = 1

  return mask

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
      torch_dtype=torch.bfloat16,
      #attn_implementation="flash_attention_2",
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
    #to_pil_image(img).save("pilled.png")
    #to_pil_image(saliency).save("sal_pilled.png")

    images_to_process = [img]
    if saliency is not None:
      images_to_process.append(saliency)
      prompt_template = prompt_dict["prompt"]
    else:
      prompt_template = prompt_dict["prompt_not_sal"]
    
    # Add the label to the prompt
    prompt_text = prompt_template.replace("{label}", label)

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
    print(f"Output txt: {output_text}")
    bboxes = parse_bboxes(output_text)
    print(f"bboxes: {bboxes} | Img shape: {img.shape[-2:]}")
    prediction = bboxes_to_mask(bboxes, img.shape[-2:])

    return prediction



def load_VLM(model_id):
  load_dotenv()
  login_to_hub()

  return VLM(model_id)