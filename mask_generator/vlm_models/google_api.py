import os
import yaml
import json
import re
import time
import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from google import genai
from google.genai import types

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

class GoogleVLMLoader:
  def __init__(self, model_id: str, prompt_path: str, sleep_time: float = 30):
    """
    model_id: e.g., 'gemini-1.5-flash' or 'gemini-1.5-pro'
    sleep_time: seconds to wait before each inference to respect free tier RPM limits
    """
    self.model_id = model_id
    self.prompt_path = prompt_path
    self.sleep_time = sleep_time

    api_key = os.getenv("GOOGLE_API_KEY")
    
    print(f"Initializing Google API Client for Model: {self.model_id}")
    self.client = genai.Client(api_key=api_key)
    
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
    qualitative: bool = False
  ):
    prompt_dict = self._load_prompt()

    if qualitative: 
      prompt_template = prompt_dict["qual_prompt"]
    else: 
      prompt_template = prompt_dict["prompt"]

    # Add the label to the prompt
    prompt_text = prompt_template.replace("{label}", label)

    # Convert tensors to PIL images
    pil_img = to_pil_image(img)
    pil_saliency = to_pil_image(saliency)

    # Build the payload
    contents = [pil_img, pil_saliency, prompt_text]

    # Sleep to prevent RateLimitError (429)
    if self.sleep_time > 0:
      print(f"Sleeping for {self.sleep_time}s to respect rate limits...")
      time.sleep(self.sleep_time)

    try:
      # Generate the response
      response = self.client.models.generate_content(
        model=self.model_id,
        contents=contents,
        config=types.GenerateContentConfig(
          max_output_tokens=512,
        )
      )
      output_text = response.text
    except Exception as e:
      print(f"API Call Failed: {e}")
      output_text = ""

    if qualitative: 
      return output_text
    
    print(f"Output txt: {output_text}")
    bboxes = parse_bboxes(output_text)
    print(f"bboxes: {bboxes} | Img shape: {img.shape[-2:]}")
    
    # Ensure we pass the image shape correctly (H, W)
    prediction = bboxes_to_mask(bboxes, img.shape[-2:])

    return prediction

def load_google_vlm(model_id: str, prompt_path: str,):
  return GoogleVLMLoader(model_id, prompt_path)