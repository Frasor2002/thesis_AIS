import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from typing import Dict, Any, Union
from torch import Tensor
from typing import Optional
from dotenv import load_dotenv
from mask_generator.utils import login_to_hub

# Qwen/Qwen3.6-27B
# Qwen/Qwen3.6-35B-A3B
# Qwen/Qwen2.5-VL-3B-Instruct
# Qwen/Qwen3-VL-2B-Instruct
# Qwen/Qwen3-VL-8B-Instruct

class VLM:
  def __init__(self, model_id: str):
    self.model_id = model_id
    
    print(f"Loading {self.model_id}")

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

  
  def detect_confounders_no_sal():
    pass

  
  def detect_confounders(
    self,
    img: Tensor,
    saliency: Optional[Tensor],
    label: str
  ):
    prompt_text = (
      f"The target classification label is '{label}'. "
      "Analyze the provided images. Find the confounders present listing them."
      "Output the name of the confounder and its bounding box based strictly on the original image. "
      "Format your answer EXACTLY as JSON: {\"confounder\": \"name\", \"bbox_2d\": [xmin, ymin, xmax, ymax]}"
    )

    content_list = []
    images_to_process = []

    if saliency is not None:
      content_list.extend([
        {"type": "text", "text": "Image 1 (Original):"},
        {"type": "image"},
        {"type": "text", "text": "Image 2 (Saliency Map):"},
        {"type": "image"}
      ])
      images_to_process.extend([img, saliency])
    else:
      content_list.extend([
          {"type": "image"}
      ])
      images_to_process.append(img)

    content_list.append({"type": "text", "text": prompt_text})

    messages = [
      {
        "role": "user",
        "content": content_list,
      }
    ]

    text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
    inputs = self.processor(
      text=[text], 
      images=images_to_process, 
      return_tensors="pt"
    ).to(self.device)

    with torch.no_grad():
      generated_ids = self.model.generate(**inputs, max_new_tokens=256)

    # 5. Trim prompt and decode
    generated_ids_trimmed = [
      out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
        
    raw_response = self.processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return raw_response.strip()



def load_VLM():
  load_dotenv()
  login_to_hub()

  

  pass