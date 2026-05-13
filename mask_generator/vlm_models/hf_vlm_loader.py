from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from typing import Optional
import os
import yaml
import json
import re

#google/gemma-4-31B-it
#google/gemma-4-26B-A4B-it
#Qwen/Qwen3.6-27B
#Qwen/Qwen3.5-9B

processor = AutoProcessor.from_pretrained("google/gemma-4-31B-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-4-31B-it")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


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
    # Clamp to make them stay inside bounds
    xmin = max(0, min(xmin, W))
    xmax = max(0, min(xmax, W))
    ymin = max(0, min(ymin, H))
    ymax = max(0, min(ymax, H))

    mask[ymin:ymax, xmin:xmax] = 1

  return mask

class VLMLoader:
  def __init__(self, model_id: str, prompt_path: str):
    self.model_id = model_id
    self.prompt_path  = prompt_path

    print(f"Loading {self.model_id}")

    self.processor = AutoProcessor.from_pretrained(
      self.model_id, 
      trust_remote_code=True
    )

    self.model = AutoModelForImageTextToText.from_pretrained(
      self.model_id,
      torch_dtype=torch.bfloat16,
      device_map="auto",
      trust_remote_code=True
    ).eval()
    
  def _load_prompt(self) -> dict:
    if not os.path.exists(self.prompt_path):
      raise FileNotFoundError(f"Prompt file not found at: {self.prompt_path}")
            
    with open(self.prompt_path, "r", encoding="utf-8") as file:
      prompt_data = yaml.safe_load(file)
            
    return prompt_data

  def detect_confounders(
    self,
    img: Tensor,
    label: str,
    saliency: Tensor,
    qualitative: bool= False
  ):
    prompt_dict = self._load_prompt()

    images_to_process = [img, saliency]
    if qualitative: prompt_template = prompt_dict["qual_prompt"]
    else: prompt_template = prompt_dict["prompt"]

    # Add the label to the prompt
    prompt_text = prompt_template.replace("{label}", label)

    # Build the message content list
    content = []
    for image in images_to_process:
      # Pass standard PIL image format in the content block
      content.append({"type": "image", "image": to_pil_image(image)})      
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]

    inputs = self.processor.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=True,
      return_dict=True,
      return_tensors="pt"
    ).to(self.model.device)

    with torch.no_grad(): 
      outputs = self.model.generate(**inputs, max_new_tokens=512)

    # Trim input prompt from the generated tokens
    generated_ids_trimmed = outputs[0][inputs["input_ids"].shape[-1]:]

    output_text = self.processor.decode(
      generated_ids_trimmed, 
      skip_special_tokens=True, 
      clean_up_tokenization_spaces=False
    ).strip()
    if qualitative: return output_text
    
    print(f"Output txt: {output_text}")
    bboxes = parse_bboxes(output_text)
    print(f"bboxes: {bboxes} | Img shape: {img.shape[-2:]}")
    prediction = bboxes_to_mask(bboxes, img.shape[-2:])

    return prediction



def load_hf_vlm(model_id: str, prompt_path: str):
  return VLMLoader(model_id, prompt_path)