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
from ...utils.model import MODALITY, get_module

class ResNet50GPTQ(BaseGPTQModel):
    is_hf_model = False
    loader = resnet50
    
    # We will now correctly handle the base modules.
    base_modules = ["conv1", "fc"]
    
    # We define the repeating block structure.
    layers_node = ["layer1", "layer2", "layer3", "layer4"]
    layer_type = "Bottleneck"
    layer_modules = [
        ["conv1"],
        ["conv2"],
        ["conv3"],
        ["downsample.0"],
    ]
    layer_modules_strict = False
    modality = [MODALITY.IMAGE]
    
    # We override the get_layers to handle the four separate layer attributes.
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        """
        This method now correctly combines the four layer stages into a single
        list for the looper to iterate over.
        """
        all_blocks = nn.ModuleList()
        for stage_name in self.layers_node:
            all_blocks.extend(getattr(model, stage_name))
        return all_blocks

    def prepare_dataset(
        self,
        calibration_dataset: Union[List[Image.Image], Dataset],
        batch_size: int = 1,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        This method now returns data in the dictionary format that the
        original, unmodified looper expects. The key 'x' will be passed
        to the model's forward method.
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
            # The model's forward method expects a single tensor, not kwargs.
            # We must use a standard key that the looper will unpack as a positional argument.
            batched_data.append({'x': batch})
            
        return batched_data