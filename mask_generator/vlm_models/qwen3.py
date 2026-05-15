import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from typing import Optional
import os
import yaml


class Qwen3VLInstructLoader:
  def __init__(self, model_id: str, prompt_path: str):
    self.model_id = model_id
    self.prompt_path  = prompt_path

    print(f"Loading {self.model_id}")

    self.processor = AutoProcessor.from_pretrained(
      self.model_id, 
      trust_remote_code=True
    )

    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16,
      device_map="auto"
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


    return output_text



def load_qwen3_vl_instruct(model_id, prompt_path: str):

  model = Qwen3VLInstructLoader(model_id, prompt_path)
  return model