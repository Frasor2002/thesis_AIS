import PIL.Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import os
import yaml
import PIL
import torch

class VLMLoader:
  def __init__(self, model_id: str, prompt_path: str):
    self.model_id = model_id
    self.prompt_path = prompt_path

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    self.processor = AutoProcessor.from_pretrained(model_id)
    self.model = AutoModelForImageTextToText.from_pretrained(
      model_id,
      quantization_config=quantization_config,
      dtype="auto",
      device_map="auto"
    )
  
  def _load_prompt(self) -> dict:
    if not os.path.exists(self.prompt_path):
      raise FileNotFoundError(f"Prompt file not found at: {self.prompt_path}")
            
    with open(self.prompt_path, "r", encoding="utf-8") as file:
      prompt_data = yaml.safe_load(file)
            
    return prompt_data["prompt"]


  def detect_confounders(self, img: PIL.Image, saliency: PIL.Image, pred: str, label: str):
    prompt = self._load_prompt()
    prompt = prompt.replace("{prediction}", pred)
    prompt = prompt.replace("{label}", label)

    messages = [
      {
        "role": "user",
        "content": [
          {"type": "image"},
          {"type": "image"},
          {"type": "text", "text": prompt},
        ]
      },
    ]
    formatted_prompt = self.processor.apply_chat_template(
      messages, 
      add_generation_prompt=True, 
      tokenize=False,
    )

    inputs = self.processor(
      text=formatted_prompt, 
      images=[img, saliency], 
      return_tensors="pt"
    ).to(self.model.device)

    with torch.no_grad():
      output = self.model.generate(
        **inputs, 
        max_new_tokens=500
      )
      
    output_text = self.processor.decode(
      output[0], 
      skip_special_tokens=True
    )

    return output_text