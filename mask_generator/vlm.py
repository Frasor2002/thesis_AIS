import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Dict, Any, Union
from torch import Tensor
from typing import Optional

#from dotenv import load_dotenv

class VLM:
  def __init__(self, model_id: str, device: str = "cuda"):
    self.model_id = model_id
    self.device = device
    self.load_model()

  def load_model(self):
    print(f"Loading {self.model_id} on {self.device}...")

    self.processor = AutoProcessor.from_pretrained(
      self.model_id, 
      trust_remote_code=True
    )
    
    self.model = AutoModelForVision2Seq.from_pretrained(
      self.model_id,
      torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
      device_map="auto",
      trust_remote_code=True
    ).eval()
  
  def detect_confounders(
    self,
    img: Tensor,
    saliency: Optional[Tensor],
    label: str
  ):
    pass
