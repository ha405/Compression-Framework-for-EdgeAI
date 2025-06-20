# auto_gptq/modeling/resnet.py

from typing import List, Dict, Union

import torch
import torch.nn as nn
from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.transforms import v2 as transforms

from ..base import BaseGPTQModel
from ...utils.data import collate_data
from ...utils.model import MODALITY


class ResNet50GPTQ(BaseGPTQModel):
    """
    GPTQ configuration for the torchvision.models.resnet50 model.
    """
    loader = resnet50
    is_hf_model = False
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
    modality = [MODALITY.IMAGE]
    require_load_processor = False

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        """
        Overrides the default layer-finding mechanism to correctly handle the
        ResNet architecture by collecting blocks from layer1, layer2, layer3, and layer4.
        """
        all_blocks = []
        all_blocks.extend(model.layer1)
        all_blocks.extend(model.layer2)
        all_blocks.extend(model.layer3)
        all_blocks.extend(model.layer4)
        return nn.ModuleList(all_blocks)

    def prepare_dataset(
        self,
        calibration_dataset: Union[List[Image], Dataset],
        batch_size: int = 1,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Overrides the base method to prepare an image dataset for calibration.

        Args:
            calibration_dataset (Union[List[Image], Dataset]): A list of PIL Images
                or a PyTorch Dataset that returns PIL images.
            batch_size (int): The batch size for calibration.

        Returns:
            A list of dictionaries, where each dictionary represents a batch
            and contains the key 'x' with a batched tensor of pixel values.
        """
        image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=True), 
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
            dataset = ImageDataset(calibration_dataset, image_transforms) 
        else:
            raise ValueError("calibration_dataset must be a list of PIL Images or a PyTorch Dataset.")
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        batched_data = []
        for batch in data_loader:
            batched_data.append({'x': batch})
            
        return batched_data