import torch
from torch.nn import CrossEntropyLoss, Module
from typing import Callable
import torch.nn.functional as F
from typing import Optional
from typing import Any

class LossWrapper(Module):
  def __init__(self, base_loss:Callable=CrossEntropyLoss()) -> None:
    """Initialize loss wrapper.
    Args:
      base_loss (Callable): loss to wrap.
    """
    super().__init__()
    self.base_loss = base_loss

  def forward(self, logits, targets, inputs=None, masks=None):
    """Forward function that executes the base logic.
    Args: 
      logits (Tensor): model logits.
      targets (Tensor): labels.
      inputs (Tensor): input images.
      masks (Tensor): explaination masks with 0 in correct areas and 1 where confounders are.
    Returns:
      Tensor: base loss.
    """
    return self.base_loss(logits, targets)  


class RRR(Module):
  def __init__(self, reg_rate:float=1, base_loss:Callable=CrossEntropyLoss(), normalize: bool=True) -> None:
    """Initialize Right for Right Reasons.
    Args:
      reg_rate (float): regularization rate.
      base_loss (Callable): base loss used for optimizing the model for being right.
      rr_clip (Optional[float]): float value to clip RR term.
    """
    super().__init__()
    self.reg_rate = reg_rate
    self.base_loss = base_loss
    self.normalize = normalize
  
  def forward(self, logits:torch.Tensor, targets: torch.Tensor, inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Compute RRR loss with mask penalty.
    Args: 
      logits (Tensor): model logits.
      targets (Tensor): labels.
      inputs (Tensor): input images.
      masks (Tensor): explaination masks with 0 in correct areas and 1 where confounders are.
    Returns:
      Tensor: RRR loss.
    """
    # Right answer loss term
    ra_loss = self.base_loss(logits, targets)

    if self.reg_rate == 0 or masks.sum() == 0:
      self.curr_ce = ra_loss.item()
      self.curr_rr = 0.0
      return ra_loss

    # Right reason loss term
    probs = F.log_softmax(logits, dim=1)
    probs.retain_grad()
    grads = torch.autograd.grad(
      outputs=probs,
      inputs=inputs,
      grad_outputs=torch.ones_like(probs),
      create_graph=True,
      allow_unused=True
    )[0]
    rr_penalty = torch.mul(masks, grads) ** 2

    if self.normalize:
      # Number of channels
      num_channels = inputs.size(1)    
      effective_mask_elements = masks.sum() * num_channels
      # Mean of attribution per pixel
      rr_loss = torch.sum(rr_penalty) / effective_mask_elements
    else:
      rr_loss = torch.sum(rr_penalty)

    rr_loss *= self.reg_rate

    self.curr_ce = ra_loss.item()
    self.curr_rr = rr_loss.item()

    final_loss = ra_loss + rr_loss

    return final_loss



def load_loss_fun(name: str, **kwargs: Any) -> Any:
  """Load loss function.
  Args:
    name (str): loss function name.
    **kwargs (Any): additional loss parameters.
  Returns:
    Any: loss function chosen.
  """

  loss_functions = {
    "CrossEntropy": LossWrapper,
    "RRR": RRR
  }
  
  if name not in loss_functions.keys():
    raise ValueError("Wrong loss function name.")
  
  loss_fun_class = loss_functions[name]
  loss_fun = loss_fun_class(**kwargs)
  return loss_fun