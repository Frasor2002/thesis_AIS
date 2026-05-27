import PIL.Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
import yaml
import torch

class VLMLoader:
  def __init__(self, model_id: str, prompt_path: str):
    self.model_id = model_id
    self.prompt_path = prompt_path

    self.processor = AutoProcessor.from_pretrained(model_id)
    self.model = AutoModelForImageTextToText.from_pretrained(
      model_id,
      torch_dtype="auto", 
      device_map="auto"
    )
    # Load both prompts into a dictionary
    self.prompts = self._load_prompts()

  def _load_prompts(self) -> dict:
    if not os.path.exists(self.prompt_path):
      raise FileNotFoundError(f"Prompt file not found at: {self.prompt_path}")
            
    with open(self.prompt_path, "r", encoding="utf-8") as file:
      prompt_data = yaml.safe_load(file)

    # Ensure these keys match your prompt.yaml exactly
    return {
      "right_reasons": prompt_data["prompt_rr"],
      "wrong_reasons": prompt_data["prompt_wr"]
    }

  # Added 'right_reasons' boolean flag here
  def detect_confounders(self, img: PIL.Image, saliency: PIL.Image, pred: str, label: str, right_reasons: bool = False):
    
    # Select the correct base prompt
    base_prompt = self.prompts["right_reasons"] if right_reasons else self.prompts["wrong_reasons"]
    
    prompt = base_prompt.replace("{prediction}", str(pred))
    prompt = prompt.replace("{label}", str(label))
    prompt = prompt.replace("{img_len}", str(img.width))

    messages = [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": img},
          {"type": "image", "image": saliency}, 
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
        max_new_tokens=512 
      )
    
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[0][prompt_length:]
    output_text = self.processor.decode(
      new_tokens, 
      skip_special_tokens=True
    ).strip()

    return output_text