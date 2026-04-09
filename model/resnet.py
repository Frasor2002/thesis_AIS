import torch
import torch.nn as nn
from typing import Optional, Type


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, padding: int = 0) -> nn.Conv2d:
  """Get a convolutional layer with kernel 3x3 and no bias.
  Args:
    in_channels (int): number of input channels.
    out_channels (int): number of output channels.
    stride (int): stride of convolution.
    padding (int): zero-padding added to both sides of input.
  Returns:
    nn.Conv2d: The 3x3 convolutional layer.
  """
  return nn.Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=3,
    stride=stride,
    padding=padding,
    bias=False
  )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, padding: int = 0) -> nn.Conv2d:
  """Get a convolutional layer with kernel 1x1 and no bias.
  Args:
    in_channels (int): number of input channels.
    out_channels (int): number of output channels.
    stride (int): stride of convolution.
    padding (int): zero-padding added to both sides of input.
  Returns:
    nn.Conv2d: The 1x1 convolutional layer.
  """
  return nn.Conv2d(
    in_channels=in_channels, 
    out_channels=out_channels, 
    kernel_size=1, 
    stride=stride, 
    padding=padding, 
    bias=False
    )


def downsample(in_channels: int, out_channels: int, expansion: int, stride: int) -> Optional[nn.Sequential]:
  """Return if needed a downsample module (1x1 conv -> batch_norm) for residual connections if dimension is mismatched.
  Args:
    in_channels (int): number of input channels.
    out_channels (int): number of output channels before expansion.
    expansion (int): expansion factor of the block (i.e. 4 for Bottleneck, 1 for BasicBlock).
    stride (int): stride of the convolution.
  Returns:
    Optional[nn.Sequential]: A downsampling module if required, else None.
  """
  if in_channels != out_channels*expansion or stride != 1:
    return nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels*expansion, kernel_size=1, stride=stride, padding=0, bias=False),
      nn.BatchNorm2d(num_features=out_channels*expansion)
    )
  return None


class BasicBlock(nn.Module):
  """Standard residual block used in ResNet18 and ResNet34. Contains two 3x3 convolutions."""

  def __init__(self, in_channels: int, mid_channels: int, expansion:int, stride: int) -> None:
    """Initialize BasicBlock.
    Args:
      in_channels (int): number of input channels.
      mid_channels (int): number of intermediate channels.
      expansion (int): expansion factor.
      stride (int): stride for first conv 3x3 layer.
    """
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_channels, mid_channels, stride, padding=1)
    self.bn1 = nn.BatchNorm2d(mid_channels)
    self.relu = nn.ReLU()
    self.conv2 = conv3x3(mid_channels, mid_channels, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(mid_channels)

    self.downsample = downsample(in_channels, mid_channels, expansion, stride)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.
    Args:
      x (Tensor): input tensor.
    Returns:
      Tensor: output tensor.
    """
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None: x = self.downsample(x)
    out += x
    out = self.relu(out)
    return out



class Bottleneck(nn.Module):
  """Bottleneck residual block used in ResNet50, ResNet101 and ResNet150.
  Contains 1x1->3x3->1x1 convolution to reduce parameter count."""

  def __init__(self, in_channels: int, mid_channels: int, expansion: int, stride: int) -> None:
    """Initialize Bottleneck.
    Args:
      in_channels (int): number of input channels.
      mid_channels (int): number of intermediate channels.
      expansion (int): expansion factor.
      stride (int): stride for conv 3x3 layer.
    """
    super(Bottleneck, self).__init__()

    self.downsample = downsample(in_channels, mid_channels, expansion, stride)
    
    self.conv1 = conv1x1(in_channels, mid_channels)
    self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
    self.relu = nn.ReLU()
    self.conv2 = conv3x3(mid_channels, mid_channels, stride, padding=1)
    self.bn2 = nn.BatchNorm2d(num_features=mid_channels)
    self.conv3 = conv1x1(mid_channels, mid_channels*expansion)
    self.bn3 = nn.BatchNorm2d(num_features=mid_channels*expansion)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.
    Args:
      x (Tensor): input tensor.
    Returns:
      Tensor: output tensor.
    """

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None: x = self.downsample(x)
    out += x
    out = self.relu(out)
    return out


class ResNet(nn.Module):
  """ResNet module."""

  def __init__(self, resnet_params: tuple, n_classes: int) -> None:
    """Initialize ResNet.
    Args:
      resnet_params (tuple): tuple containing configuration for the network.
      n_classes (int): number of classes to predict or output channels.
    """
    super(ResNet, self).__init__()

    self.channels_list = resnet_params[0]
    self.repetition_list = resnet_params[1]
    self.expansion = resnet_params[2]
    self.is_bottleneck = resnet_params[3]
    self.in_channels = 3 # RGB image
    self.n_classes = n_classes
    self.start_channels = 64

    # Common input layer to every ResNet variant
    self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.start_channels, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(self.start_channels)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)


    # Four intermediate block
    self.layer1 = self._make_block(repetitions=self.repetition_list[0], in_channels=64, mid_channels=self.channels_list[0], expansion=self.expansion, stride=1, is_bottleneck=self.is_bottleneck)
    self.layer2 = self._make_block(repetitions=self.repetition_list[1], in_channels=self.channels_list[0]*self.expansion, mid_channels=self.channels_list[1], expansion=self.expansion, stride=2, is_bottleneck=self.is_bottleneck)
    self.layer3 = self._make_block(repetitions=self.repetition_list[2], in_channels=self.channels_list[1]*self.expansion, mid_channels=self.channels_list[2], expansion=self.expansion, stride=2, is_bottleneck=self.is_bottleneck)
    self.layer4 = self._make_block(repetitions=self.repetition_list[3], in_channels=self.channels_list[2]*self.expansion, mid_channels=self.channels_list[3], expansion=self.expansion, stride=2, is_bottleneck=self.is_bottleneck)

    # Common output layer
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(in_features=self.channels_list[3]*self.expansion, out_features=self.n_classes)

    # Init weights
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
  
  

  def _make_block(self, repetitions: int, in_channels: int, mid_channels: int, expansion: int, stride: int, is_bottleneck: bool) -> nn.Sequential:
    """Stack multiple residual blocks.
    Args:
      repetitions (int): number of blocks to stack.
      in_channels (int): number of input channels for the first block.
      mid_channels (int): number of ouput channels without expansion.
      expansion (int): expansion factor.
      stride (int): stride for convolutions inside the residual blocks.
      is_bottleneck (bool): boolean to tell if BasicBlock or Bottleneck needs to be used.
    Returns:
      Sequential: sequential module stacking all residual blocks.    
    """
    
    def get_class(is_bottleneck: bool) -> Type[nn.Module]:
      """Return a residual block class.
      Args:
        is_bottleneck (bool): boolean to tell if BasicBlock or Bottleneck needs to be used.
      Returns:
        Type[nn.Module]: a residual block.
      """
      if is_bottleneck: return Bottleneck
      else: return BasicBlock
    
    layers = []
    resblock = get_class(is_bottleneck)
    # Input layer
    layers.append(resblock(in_channels=in_channels, mid_channels=mid_channels, expansion=expansion, stride=stride))
    # Repeated layers repetitions -1
    for _ in range(1, repetitions):
      layers.append(resblock(in_channels=mid_channels*expansion, mid_channels=mid_channels, expansion=expansion, stride=1))

    return nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.
    Args:
      x (Tensor): input tensor.
    Returns:
      Tensor: output tensor.
    """

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.maxpool(out)

    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    out = self.avgpool(out)
    out = torch.flatten(out, start_dim=1)
    out = self.fc(out)
    return out
    

def load_resnet(model_name: str, n_classes: int, device: str = "cpu") -> ResNet:
  """Load a specific ResNet version given name and number of classes for classification.
  Args:
    name (str): name of the ResNet.
    n_classes (int): number of output classes.
    device (str): device to load the model on.
  Returns:
    ResNet: ResNet model chosen.
  """

  # resnetX = (Num of channels, repetition, Bottleneck_expansion , Bottleneck_layer)
  model_param={
    'resnet18': ([64,128,256,512],[2,2,2,2],1,False),
    'resnet34': ([64,128,256,512],[3,4,6,3],1,False),
    'resnet50': ([64,128,256,512],[3,4,6,3],4,True),
    'resnet101': ([64,128,256,512],[3,4,23,3],4,True),
    'resnet152': ([64,128,256,512],[3,8,36,3],4,True)
  }

  if model_name not in model_param.keys():
    raise ValueError("Wrong ResNet version name.")
  
  model = ResNet(model_param[model_name], n_classes)
  model = model.to(device)

  return model
  
