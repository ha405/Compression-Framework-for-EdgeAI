from typing import List, Dict, Union, Tuple

import torch
import torch.nn as nn
from PIL.Image import Image
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
    
    base_modules = []

    pre_lm_head_norm_module = None

    layers_node = None
    
    layer_type = "Bottleneck"
    
    layer_modules = [
        ["conv1"],
        ["conv2"],
        ["conv3"],
        ["downsample.0"],
    ]
    
    layer_modules_strict = False
    
    modality = [MODALITY.IMAGE]
    
    require_load_processor = False

    def get_layers(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        named_blocks = []
        
        all_quantizable_layers = find_modules(model, [nn.Conv2d, nn.Linear])
        
        layers_to_quantize = []
        for name, module in all_quantizable_layers.items():
            if name == "conv1" or name == "fc":
                continue
            layers_to_quantize.append((name, module))
            
        return layers_to_quantize

    def prepare_dataset(
        self,
        calibration_dataset: Union[List[Image], Dataset],
        batch_size: int = 1,
        **kwargs
    ) -> List[torch.Tensor]:
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

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = self.images[idx]
                if not isinstance(img, Image.Image):
                    img = transforms.ToPILImage()(img)
                return self.transform(img)

        if isinstance(calibration_dataset, list):
            dataset = ImageDataset(calibration_dataset, image_transforms)
        elif isinstance(calibration_dataset, Dataset):
            dataset = ImageDataset(calibration_dataset, image_transforms)
        else:
            raise ValueError("calibration_dataset must be a list of PIL Images or a PyTorch Dataset.")
            
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        batched_data = []
        for batch in data_loader:
            batched_data.append(batch)
            
        return batched_data