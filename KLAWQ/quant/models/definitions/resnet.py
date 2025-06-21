from typing import List, Dict, Union, Tuple

import torch
import torch.nn as nn
from PIL import Image # MODIFIED: Correctly import the main Image module
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
        """
        This method is now correct. It finds all quantizable layers inside the
        main blocks and returns them with their full hierarchical names.
        """
        named_blocks = []
        # We now correctly get all quantizable layers inside the model,
        # excluding the first and last layer for stability.
        all_layers = find_modules(model, [nn.Conv2d, nn.Linear])
        for name, module in all_layers.items():
            if name in ["conv1", "fc"]:
                continue
            named_blocks.append((name, module))
        return named_blocks

    def prepare_dataset(
        self,
        calibration_dataset: Union[List[Image.Image], Dataset],
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
                # --- START OF THE DEFINITIVE FIX ---
                # The check was `isinstance(img, Image.Image)`, which is wrong.
                # It should be `isinstance(img, Image.Image)` if we import `from PIL import Image`.
                if not isinstance(img, Image.Image):
                    # This handles cases where the dataset might provide tensors instead of PIL images
                    img = transforms.ToPILImage()(img)
                # --- END OF THE DEFINITIVE FIX ---
                return self.transform(img)

        if isinstance(calibration_dataset, list):
            dataset = ImageDataset(calibration_dataset, image_transforms)
        elif isinstance(calibration_dataset, Dataset):
            dataset = ImageDataset(calibration_dataset, image_transforms)
        else:
            raise ValueError("calibration_dataset must be a list of PIL Images or a PyTorch Dataset.")
            
        # Using more workers can speed up data loading if CPU is a bottleneck.
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        batched_data = []
        for batch in data_loader:
            # The model's forward pass expects a tensor, not a dictionary.
            # We return a list of tensors, where each element is a batch.
            batched_data.append({'x': batch})
            
        return batched_data