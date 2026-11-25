from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class PatchCoreLite(nn.Module):
    def __init__(
        self,
        bank: Tensor | None = None,
        projection_seed: int = 42,
        output_size: int = 256,
    ) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.output_size = output_size

        generator = torch.Generator().manual_seed(projection_seed)
        projection = torch.randn((384, 128), generator=generator) / math.sqrt(128)
        self.register_buffer("projection", projection)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("bank", bank if bank is not None else torch.empty((0, 128)))
        self.register_buffer("bank_sq_norm", self.bank.square().sum(dim=1))
        self.register_buffer("gaussian", self._gaussian_kernel())
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _gaussian_kernel(size: int = 11, sigma: float = 2.0) -> Tensor:
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel_1d = torch.exp(-(coords.square()) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = torch.outer(kernel_1d, kernel_1d)
        return kernel.view(1, 1, size, size)

    @torch.no_grad()
    def embed(self, image: Tensor) -> Tensor:
        x = (image - self.mean) / self.std
        x = self.stem(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        combined = torch.cat((layer2, layer3), dim=1)
        return F.conv2d(combined, self.projection.t().unsqueeze(-1).unsqueeze(-1))

    @torch.no_grad()
    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        if self.bank.shape[0] == 0:
            raise RuntimeError("PatchCore memory bank is empty")
        embeddings = self.embed(image).flatten(2).transpose(1, 2)
        query_sq_norm = embeddings.square().sum(dim=2, keepdim=True)
        distances = query_sq_norm + self.bank_sq_norm.view(1, 1, -1) - 2 * torch.matmul(
            embeddings, self.bank.t()
        )
        anomaly = distances.clamp_min(0).amin(dim=2).view(image.shape[0], 1, -1)
        anomaly = anomaly.view(image.shape[0], 1, *self._feature_shape(image))
        anomaly = F.interpolate(
            anomaly, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
        )
        anomaly = F.conv2d(anomaly, self.gaussian, padding=5)
        return anomaly, anomaly.flatten(1).amax(dim=1)

    @staticmethod
    def _feature_shape(image: Tensor) -> tuple[int, int]:
        return (image.shape[-2] // 8, image.shape[-1] // 8)


def greedy_coreset(features: Tensor, count: int, seed: int = 42) -> Tensor:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D tensor")
    if count <= 0 or count > features.shape[0]:
        raise ValueError("count must be between 1 and the number of feature rows")

    generator = torch.Generator(device=features.device).manual_seed(seed)
    first = torch.randint(features.shape[0], (1,), generator=generator, device=features.device).item()
    selected_indices = torch.empty(count, dtype=torch.long, device=features.device)
    selected_indices[0] = first
    feature_norms = features.square().sum(dim=1)
    current = features[first]
    min_distances = (feature_norms + current.square().sum() - 2 * (features @ current)).clamp_min(0)

    for index in range(1, count):
        next_index = torch.argmax(min_distances)
        selected_indices[index] = next_index
        current = features[next_index]
        distances = (feature_norms + current.square().sum() - 2 * (features @ current)).clamp_min(0)
        min_distances = torch.minimum(min_distances, distances)

    return features.index_select(0, selected_indices)
