# auto_gptq/modeling/resnet.py

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
# Make sure this import is correct for your project structure
from ...utils.model import MODALITY, get_module


class ResNet50GPTQ(BaseGPTQModel):
    """
    GPTQ configuration for the torchvision.models.resnet50 model.
    """
    is_hf_model = False
    loader = resnet50
    base_modules = [
        "conv1",
        "fc",
    ]
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
        """
        Corrected layer-finding mechanism.
        This now returns ALL quantizable layers: the base modules (conv1, fc)
        and all the bottleneck blocks from the main stages.
        """
        all_quantizable_layers = []

        # 1. Add the standalone base layers (conv1, fc)
        for name in self.base_modules:
            module = get_module(model, name)
            if module:
                all_quantizable_layers.append((name, module))

        # 2. Add the blocks from the main stages (layer1, layer2, etc.)
        for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
            stage = getattr(model, stage_name)
            for i, block in enumerate(stage):
                all_quantizable_layers.append((f"{stage_name}.{i}", block))
        
        return all_quantizable_layers

    def prepare_dataset(
        self,
        calibration_dataset: Union[List[Image], Dataset],
        batch_size: int = 1,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Overrides the base method to prepare an image dataset for calibration.
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

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = self.images[idx]
                return self.transform(img)

        if isinstance(calibration_dataset, list):
            dataset = ImageDataset(calibration_dataset, image_transforms)
        elif isinstance(calibration_dataset, Dataset):
            # This assumes the dataset yields PIL Images
            dataset = ImageDataset([img for img in calibration_dataset], image_transforms)
        else:
            raise ValueError("calibration_dataset must be a list of PIL Images or a PyTorch Dataset.")
            
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        batched_data = []
        for batch in data_loader:
            # The model expects the input tensor directly, not in a dict with key 'x'
            batched_data.append({'x': batch})
            
        return batched_data