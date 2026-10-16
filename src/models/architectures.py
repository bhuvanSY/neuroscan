"""PyTorch Model Architectures for NeuroScan AI.

Three Independent Architectures Trained Strictly from Scratch:
1. MODEL A: Custom CNN (Progressive channel expansion, Conv-Norm-Act blocks, Pooling, Dropout, AdaptiveAvgPool)
2. MODEL B: ResNet-50 (Standard ResNet-50 initialized from scratch with random parameters, 4-class head)
3. MODEL C: DenseNet-121 (Standard DenseNet-121 initialized from scratch with random parameters, 4-class head)

GUARANTEE:
Never loads ImageNet weights, pretrained checkpoints, or pretrained feature extractors.
Every learnable parameter starts from random initialization.
"""

import math
from typing import Dict, Type
import torch
import torch.nn as nn
from torchvision.models import resnet50, densenet121, efficientnet_v2_s


def initialize_weights(module: nn.Module) -> None:
    """Apply Kaiming normal initialization across convolutional and linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)


# =========================================================================
# MODEL A: Custom CNN
# =========================================================================
class ConvNormActBlock(nn.Module):
    """Reusable Conv2D - BatchNorm2D - LeakyReLU - Dropout block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout_p: float = 0.15,
        pool: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """Deep Custom CNN with progressive feature-channel expansion (32 -> 64 -> 128 -> 256 -> 512).
    
    Trained strictly from scratch with Kaiming initialization.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 4, base_channels: int = 32):
        super().__init__()
        self.stage1 = ConvNormActBlock(in_channels, base_channels, dropout_p=0.10, pool=True)           # 32 channels
        self.stage2 = ConvNormActBlock(base_channels, base_channels * 2, dropout_p=0.15, pool=True)     # 64 channels
        self.stage3 = ConvNormActBlock(base_channels * 2, base_channels * 4, dropout_p=0.20, pool=True) # 128 channels
        self.stage4 = ConvNormActBlock(base_channels * 4, base_channels * 8, dropout_p=0.25, pool=True) # 256 channels
        self.stage5 = ConvNormActBlock(base_channels * 8, base_channels * 16, dropout_p=0.30, pool=True) # 512 channels

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(base_channels * 16, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        # Explicit random initialization
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits


# =========================================================================
# MODEL B: ResNet-50 (From Scratch)
# =========================================================================
def build_resnet50_scratch(num_classes: int = 4, in_channels: int = 3) -> nn.Module:
    """Instantiate standard ResNet-50 with 100% randomly initialized weights.
    
    Zero pretrained weights loaded.
    """
    # Explicitly weights=None ensures no pretrained checkpoint is fetched
    model = resnet50(weights=None)
    
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    # Replace final classification head with 4-class output
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    # Explicitly re-initialize all parameters randomly
    initialize_weights(model)
    return model


# =========================================================================
# MODEL C: DenseNet-121 (From Scratch)
# =========================================================================
def build_densenet121_scratch(num_classes: int = 4, in_channels: int = 3) -> nn.Module:
    """Instantiate standard DenseNet-121 with 100% randomly initialized weights.
    
    Zero pretrained weights loaded.
    """
    # Explicitly weights=None ensures no pretrained checkpoint is fetched
    model = densenet121(weights=None)
    
    if in_channels != 3:
        model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    # Replace final classification head with 4-class output
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    # Explicitly re-initialize all parameters randomly
    initialize_weights(model)
    return model


# =========================================================================
# MODEL D: EfficientNetV2-S (From Scratch)
# =========================================================================
def build_efficientnet_v2_scratch(num_classes: int = 4, in_channels: int = 3) -> nn.Module:
    """Instantiate standard EfficientNetV2-S with 100% randomly initialized weights.
    
    Zero pretrained weights loaded.
    """
    # Explicitly weights=None ensures no pretrained checkpoint is fetched
    model = efficientnet_v2_s(weights=None)
    
    if in_channels != 3:
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        
    # Replace final classification head with 4-class output
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    # Explicitly re-initialize all parameters randomly
    initialize_weights(model)
    return model


# Backward compatibility alias
NeuroConvNet = CustomCNN


# =========================================================================
# Model Registry & Factory
# =========================================================================
MODEL_BUILDERS = {
    "custom_cnn": lambda nc, ic: CustomCNN(in_channels=ic, num_classes=nc),
    "neuro_convnet": lambda nc, ic: CustomCNN(in_channels=ic, num_classes=nc),
    "resnet50": lambda nc, ic: build_resnet50_scratch(num_classes=nc, in_channels=ic),
    "densenet121": lambda nc, ic: build_densenet121_scratch(num_classes=nc, in_channels=ic),
    "efficientnet_v2": lambda nc, ic: build_efficientnet_v2_scratch(num_classes=nc, in_channels=ic),
    "efficientnet_v2_s": lambda nc, ic: build_efficientnet_v2_scratch(num_classes=nc, in_channels=ic),
}


def create_model(model_name: str, in_channels: int = 3, num_classes: int = 4) -> nn.Module:
    """Instantiate a requested model architecture strictly from scratch.
    
    Args:
        model_name: One of 'custom_cnn', 'resnet50', 'densenet121', 'efficientnet_v2' (or 'neuro_convnet')
        in_channels: Number of image channels (default 3)
        num_classes: Number of classification targets (default 4)
        
    Returns:
        Uninitialized PyTorch model with random weights.
    """
    key = model_name.lower().strip()
    if key not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model name '{model_name}'. Available scratch models: {list(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[key](num_classes, in_channels)

