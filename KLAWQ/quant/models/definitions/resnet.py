from typing import List, Dict, Union, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
try:
    from torchvision.transforms import v2 as transforms
except ImportError:
    from torchvision import transforms

from ..base import BaseGPTQModel
from ...utils.model import MODALITY, find_modules, get_module

class ResNet50GPTQ(BaseGPTQModel):
    is_hf_model = False
    loader = resnet50
    
    # DEFINITIVE FIX: Define the base modules to be quantized separately.
    # The framework has dedicated logic for these.
    base_modules = ["conv1", "fc"]
    
    # DEFINITIVE FIX: Define the four separate layer blocks. The framework
    # will iterate through these lists of blocks.
    layers_node = ["layer1", "layer2", "layer3", "layer4"]
    layer_type = "Bottleneck"
    
    # Define quantizable modules within a Bottleneck block.
    layer_modules = [
        ["conv1"],
        ["conv2"],
        ["conv3"],
        ["downsample.0"],
    ]
    
    # This remains crucial for handling blocks that don't have a downsample layer.
    layer_modules_strict = False
    
    modality = [MODALITY.IMAGE]
    
    # DEFINITIVE FIX: We NO LONGER need a custom get_layers method.
    # The configuration above is now sufficient for the original framework logic.
    
    def prepare_dataset(
        self,
        calibration_dataset: Union[List[Image.Image], Dataset],
        batch_size: int = 1,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        This method must return a list of dictionaries, as the original looper
        unpacks it with **. The key 'x' is arbitrary but conventional.
        """
        image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        class ImageDataset(Dataset):
            def __init__(self, images, transform):
                self.images = images
                self.transform = transform
            def __len__(self): return len(self.images)
            def __getitem__(self, idx):
                img = self.images[idx]
                if not isinstance(img, Image.Image):
                    img = transforms.ToPILImage()(img)
                return self.transform(img)

        dataset = ImageDataset(calibration_dataset, image_transforms)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        batched_data = []
        for batch in data_loader:
            # The model's forward method expects `model(x=...)` when using **.
            batched_data.append({'x': batch})
            
        return batched_data