from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
import os
import yaml
import json
import re


class VLMLoader:
  def __init__(self, model_id: str, prompt_path: str):
    self.model_id = model_id
    self.prompt_path  = prompt_path

    print(f"Loading {self.model_id}")

    self.processor = AutoProcessor.from_pretrained(
      self.model_id, 
      trust_remote_code=True
    )

    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4"
    )

    self.model = AutoModelForImageTextToText.from_pretrained(
      self.model_id,
      dtype=torch.bfloat16,
      device_map="auto",
      trust_remote_code=True,
      quantization_config=quantization_config,
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
    saliency: Tensor
  ):
    prompt_dict = self._load_prompt()

    images_to_process = [img, saliency]
    prompt_template = prompt_dict["prompt"]

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

    print(f"Output txt {output_text}")

    return output_text



def load_hf_vlm(model_id: str, prompt_path: str):
  return VLMLoader(model_id, prompt_path)