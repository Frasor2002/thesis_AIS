import PIL.Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
import yaml
import PIL
import torch

class VLMLoader:
  def __init__(self, model_id: str, prompt_path: str):
    self.model_id = model_id
    self.prompt_path = prompt_path

    #quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    self.processor = AutoProcessor.from_pretrained(model_id)
    self.model = AutoModelForImageTextToText.from_pretrained(
      model_id,
      dtype="auto",
      device_map="auto"
    )
  
  def _load_prompt(self) -> dict:
    if not os.path.exists(self.prompt_path):
      raise FileNotFoundError(f"Prompt file not found at: {self.prompt_path}")
            
    with open(self.prompt_path, "r", encoding="utf-8") as file:
      prompt_data = yaml.safe_load(file)
            
    return prompt_data["test"]


  def detect_confounders(self, img: PIL.Image, saliency: PIL.Image, pred: str, label: str):
    prompt = self._load_prompt()
    prompt = prompt.replace("{prediction}", pred)
    prompt = prompt.replace("{label}", label)

    messages = [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": img},
          {"type": "image", "image": img},
          {"type": "text", "text": prompt},
        ]
      },
    ]
    inputs = self.processor.apply_chat_template(
      messages, 
      add_generation_prompt=True, 
      tokenize=True,           
      return_dict=True,
      return_tensors="pt"
    ).to(self.model.device)

    with torch.no_grad():
      output = self.model.generate(
        **inputs, 
        max_new_tokens=200
      )
    
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[0][prompt_length:]
    output_text = self.processor.decode(
      new_tokens, 
      skip_special_tokens=True
    ).strip()

    return output_text