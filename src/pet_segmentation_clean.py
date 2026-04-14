"""Production-ready semantic segmentation pipeline for the Oxford-IIIT Pet dataset."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

try:
    import torchvision
except ImportError:
    torchvision = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_ROOT / "images"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
TRIMAP_DIR = ANNOTATIONS_DIR / "trimaps"
TRAINVAL_SPLIT_PATH = ANNOTATIONS_DIR / "trainval.txt"
TEST_SPLIT_PATH = ANNOTATIONS_DIR / "test.txt"
BEST_FCN_MODEL_PATH = Path("best_fcn_model.pth")
BEST_UNET_MODEL_PATH = Path("best_unet_model.pth")
BEST_AUG_UNET_MODEL_PATH = Path("best_aug_unet_model.pth")
BEST_TRANSFER_MODEL_PATH = Path("best_transfer_model.pth")
YOLO_MODEL_PATH = Path("yolov8n.pt")
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(slots=True)
class DataPaths:
    data_root: Path = DATA_ROOT
    image_dir: Path = field(init=False)
    annotations_dir: Path = field(init=False)
    trimap_dir: Path = field(init=False)
    trainval_path: Path = field(init=False)
    test_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.image_dir = self.data_root / "images"
        self.annotations_dir = self.data_root / "annotations"
        self.trimap_dir = self.annotations_dir / "trimaps"
        self.trainval_path = self.annotations_dir / "trainval.txt"
        self.test_path = self.annotations_dir / "test.txt"


@dataclass(slots=True)
class RuntimeConfig:
    batch_size: int = 16
    image_size: Tuple[int, int] = (224, 224)
    learning_rate: float = 1e-3
    num_epochs: int = 15
    transfer_phase1_epochs: int = 5
    transfer_phase2_epochs: int = 3
    num_workers: int = 0


def require_dependency(module: object, package_name: str, feature_name: str) -> None:
    if module is None:
        raise ImportError(f"Install '{package_name}' to use {feature_name}.")


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "cuda":
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_split(txt_path: Path) -> List[str]:
    file_names: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as handle:
        for line in handle:
            file_names.append(line.strip().split()[0])
    return file_names


def collect_split_paths(paths: DataPaths, split_filename: str) -> Tuple[List[Path], List[Path]]:
    split_path = paths.annotations_dir / split_filename
    image_paths: List[Path] = []
    mask_paths: List[Path] = []
    with open(split_path, "r", encoding="utf-8") as handle:
        for line in handle:
            sample_name = line.strip().split()[0]
            image_paths.append(paths.image_dir / f"{sample_name}.jpg")
            mask_paths.append(paths.trimap_dir / f"{sample_name}.png")
    return image_paths, mask_paths


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image_array = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image_array = image_array * DEFAULT_STD + DEFAULT_MEAN
    return np.clip(image_array, 0, 1)


# Preserved metric logic from the original notebook export.
def calculate_metrics(preds, targets, num_classes=2, ignore_index=255):
    pred_classes = torch.argmax(preds, dim=1)
    valid_mask = (targets != ignore_index)
    pred_valid = pred_classes[valid_mask]
    target_valid = targets[valid_mask]
    iou_list = []
    dice_list = []
    for cls in range(num_classes):
        pred_inds = (pred_valid == cls)
        target_inds = (target_valid == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou_list.append(float('nan'))
            dice_list.append(float('nan'))
        else:
            iou = intersection / union
            dice = (2.0 * intersection) / (pred_inds.sum().item() + target_inds.sum().item())
            iou_list.append(iou)
            dice_list.append(dice)
    valid_iou = [x for x in iou_list if not np.isnan(x)]
    valid_dice = [x for x in dice_list if not np.isnan(x)]
    miou = sum(valid_iou) / len(valid_iou) if valid_iou else 0.0
    mdice = sum(valid_dice) / len(valid_dice) if valid_dice else 0.0
    return miou, mdice


def calculate_iou_for_image(pred_mask: torch.Tensor, true_mask: torch.Tensor, ignore_index: int = 255) -> float:
    valid_pixels = (true_mask != ignore_index)
    pred_valid = pred_mask[valid_pixels]
    true_valid = true_mask[valid_pixels]
    intersection = torch.logical_and(pred_valid == 1, true_valid == 1).sum().item()
    union = torch.logical_or(pred_valid == 1, true_valid == 1).sum().item()
    return 0.0 if union == 0 else intersection / union


def calculate_single_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    if union == 0:
        return 1.0
    return (intersection / union).item()


def count_model_parameters(model: nn.Module) -> str:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return f"{total_parameters / 1_000_000:.2f} Million"


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


# Preserved PetDataset logic with Path-based path handling.
class PetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_name = self.file_list[idx]
        image_path = self.image_dir / f"{image_name}.jpg"
        mask_path = self.mask_dir / f"{image_name}.png"
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        mask = mask.resize((224, 224), Image.Resampling.NEAREST)
        mask_np = np.array(mask)
        new_mask = np.zeros_like(mask_np)
        new_mask[mask_np == 1] = 1
        new_mask[mask_np == 2] = 0
        new_mask[mask_np == 3] = 255
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(new_mask).long()
        return image_tensor, mask_tensor


class AugmentedPetDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], mask_paths: Sequence[Path], transform=None):
        self.image_paths = [Path(path) for path in image_paths]
        self.mask_paths = [Path(path) for path in mask_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]))
        mask[mask == 2] = 0
        mask[mask == 3] = 255
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, torch.tensor(mask, dtype=torch.long)

class SimpleFCN(nn.Module):
    def __init__(self, n_class: int = 2):
        super().__init__()
        self.e1 = self.conv_block(3, 64)
        self.e2 = self.conv_block(64, 128)
        self.e3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(256, n_class, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

    @staticmethod
    def conv_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.e1(x))
        x = self.pool(self.e2(x))
        x = self.pool(self.e3(x))
        return self.upsample(self.out_conv(x))


class UNet(nn.Module):
    def __init__(self, n_class: int = 2):
        super().__init__()
        self.enc1 = self.double_conv(3, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.double_conv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(128, 64)
        self.out_conv = nn.Conv2d(64, n_class, kernel_size=1)

    @staticmethod
    def double_conv(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.dec4(torch.cat([self.upconv4(x), e4], dim=1))
        x = self.dec3(torch.cat([self.upconv3(x), e3], dim=1))
        x = self.dec2(torch.cat([self.upconv2(x), e2], dim=1))
        x = self.dec1(torch.cat([self.upconv1(x), e1], dim=1))
        return self.out_conv(x)


class ResNet18UNet(nn.Module):
    def __init__(self, pretrained: bool = True, n_class: int = 2):
        super().__init__()
        require_dependency(smp, "segmentation-models-pytorch", "ResNet18 U-Net creation")
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=n_class,
        )

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def create_standard_loaders(paths: DataPaths, config: RuntimeConfig):
    train_list = read_split(paths.trainval_path)
    test_list = read_split(paths.test_path)
    train_dataset = PetDataset(paths.image_dir, paths.trimap_dir, train_list)
    test_dataset = PetDataset(paths.image_dir, paths.trimap_dir, test_list)
    train_loader = make_dataloader(train_dataset, config.batch_size, True, config.num_workers)
    test_loader = make_dataloader(test_dataset, config.batch_size, False, config.num_workers)
    return train_loader, test_loader, train_list, test_list


def build_augmentation_transforms(config: RuntimeConfig):
    require_dependency(A, "albumentations", "augmentation pipelines")
    require_dependency(ToTensorV2, "albumentations", "tensor conversion")
    height, width = config.image_size
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(height, width), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
        A.Normalize(mean=DEFAULT_MEAN.tolist(), std=DEFAULT_STD.tolist()),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=DEFAULT_MEAN.tolist(), std=DEFAULT_STD.tolist()),
        ToTensorV2(),
    ])
    return train_transform, test_transform


def create_augmented_loaders(paths: DataPaths, config: RuntimeConfig):
    train_transform, test_transform = build_augmentation_transforms(config)
    train_images, train_masks = collect_split_paths(paths, "trainval.txt")
    test_images, test_masks = collect_split_paths(paths, "test.txt")
    train_dataset = AugmentedPetDataset(train_images, train_masks, transform=train_transform)
    test_dataset = AugmentedPetDataset(test_images, test_masks, transform=test_transform)
    train_loader = make_dataloader(train_dataset, config.batch_size, True, config.num_workers)
    test_loader = make_dataloader(test_dataset, config.batch_size, False, config.num_workers)
    return train_loader, test_loader


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device, description: str):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    total_mdice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc=description):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            miou, mdice = calculate_metrics(outputs.cpu(), masks.cpu())
            total_loss += loss.item()
            total_miou += miou
            total_mdice += mdice
    return {
        "loss": total_loss / len(data_loader),
        "miou": total_miou / len(data_loader),
        "mdice": total_mdice / len(data_loader),
    }


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, num_epochs: int, checkpoint_path: Path, train_description: str, val_description: str, initial_best_miou: float = 0.0):
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_mdice": []}
    best_miou = initial_best_miou
    for epoch in range(num_epochs):
        print(f"\n{'=' * 15} Epoch {epoch + 1}/{num_epochs} {'=' * 15}")
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=train_description):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        average_train_loss = train_loss / len(train_loader)
        validation = evaluate_model(model, val_loader, criterion, device, val_description)
        history["train_loss"].append(average_train_loss)
        history["val_loss"].append(validation["loss"])
        history["val_miou"].append(validation["miou"])
        history["val_mdice"].append(validation["mdice"])
        print(f"Train Loss: {average_train_loss:.4f} | Val Loss: {validation['loss']:.4f}")
        print(f"Val mIoU: {validation['miou'] * 100:.2f}% | Val mDice: {validation['mdice'] * 100:.2f}%")
        if validation["miou"] > best_miou:
            best_miou = validation["miou"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint updated: '{checkpoint_path}'")
    return history, best_miou


def merge_histories(*histories: Dict[str, List[float]]) -> Dict[str, List[float]]:
    merged = {"train_loss": [], "val_loss": [], "val_miou": [], "val_mdice": []}
    for history in histories:
        for key in merged:
            merged[key].extend(history.get(key, []))
    return merged


def train_transfer_model(train_loader: DataLoader, val_loader: DataLoader, config: RuntimeConfig, device: torch.device):
    model = ResNet18UNet(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    optimizer_phase1 = optim.Adam(model.decoder.parameters(), lr=1e-3)
    phase1_history, best_miou = train_model(model, train_loader, val_loader, optimizer_phase1, criterion, device, config.transfer_phase1_epochs, BEST_TRANSFER_MODEL_PATH, "Transfer Phase 1 Train", "Transfer Phase 1 Validation")
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-5)
    phase2_history, _ = train_model(model, train_loader, val_loader, optimizer_phase2, criterion, device, config.transfer_phase2_epochs, BEST_TRANSFER_MODEL_PATH, "Transfer Phase 2 Train", "Transfer Phase 2 Validation", initial_best_miou=best_miou)
    return model, merge_histories(phase1_history, phase2_history)


def smoke_test_models(device: torch.device, image_size: Tuple[int, int]) -> None:
    height, width = image_size
    dummy_input = torch.randn(16, 3, height, width, device=device)
    print(f"Input tensor shape: {tuple(dummy_input.shape)}")
    print(f"SimpleFCN output shape: {tuple(SimpleFCN(n_class=2).to(device)(dummy_input).shape)}")
    print(f"UNet output shape: {tuple(UNet(n_class=2).to(device)(dummy_input).shape)}")
    if smp is not None:
        print(f"ResNet18 U-Net output shape: {tuple(ResNet18UNet(pretrained=False, n_class=2).to(device)(dummy_input).shape)}")


def preview_mask_conversion(paths: DataPaths, sample_name: str = "Abyssinian_1") -> None:
    image = Image.open(paths.image_dir / f"{sample_name}.jpg").convert("RGB")
    mask = Image.open(paths.trimap_dir / f"{sample_name}.png")
    mask_np = np.array(mask)
    converted_mask = mask_np.copy()
    converted_mask[mask_np == 2] = 0
    converted_mask[mask_np == 3] = 255
    converted_mask[mask_np == 1] = 1
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_title("Original image")
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[1].set_title("Remapped mask")
    axes[1].imshow(converted_mask, cmap="viridis")
    axes[1].axis("off")
    axes[2].set_title("Overlay")
    axes[2].imshow(image)
    axes[2].imshow(np.ma.masked_where(converted_mask == 255, converted_mask), alpha=0.5, cmap="autumn")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()
    print(f"Mask shape: {converted_mask.shape}")
    print(f"Unique remapped values: {np.unique(converted_mask)}")


def preview_dataloader_batch(train_loader: DataLoader) -> None:
    images, masks = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Image from DataLoader")
    plt.imshow(images[0].permute(1, 2, 0))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Mask from DataLoader")
    plt.imshow(masks[0], cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def analyze_class_balance(train_loader: DataLoader) -> None:
    images, masks = next(iter(train_loader))
    foreground_pixels = (masks == 1).sum().item()
    background_pixels = (masks == 0).sum().item()
    total_valid_pixels = foreground_pixels + background_pixels
    ratio = foreground_pixels / total_valid_pixels
    print("--- Class balance analysis ---")
    print(f"Foreground pixels: {foreground_pixels}")
    print(f"Background pixels: {background_pixels}")
    print(f"Foreground ratio: {ratio * 100:.2f}%")
    print("Warning: class imbalance detected." if ratio < 0.20 else "Class distribution is within a reasonable range.")
    plt.figure(figsize=(12, 12))
    for index in range(min(16, images.size(0))):
        plt.subplot(4, 4, index + 1)
        plt.imshow(images[index].permute(1, 2, 0).numpy())
        plt.imshow(np.ma.masked_where(masks[index].numpy() == 255, masks[index].numpy()), alpha=0.4, cmap="jet")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def analyze_trimap_statistics(train_loader: DataLoader) -> None:
    images, masks = next(iter(train_loader))
    total_pixels = masks.numel()
    foreground_pixels = (masks == 1).sum().item()
    background_pixels = (masks == 0).sum().item()
    border_pixels = (masks == 255).sum().item()
    print("--- Trimap statistics ---")
    print(f"Foreground (animal): {foreground_pixels / total_pixels * 100:.2f}%")
    print(f"Background: {background_pixels / total_pixels * 100:.2f}%")
    print(f"Ambiguous boundary: {border_pixels / total_pixels * 100:.2f}%")
    sample_count = min(3, images.size(0))
    figure, axes = plt.subplots(sample_count, 2, figsize=(10, 4 * sample_count))
    if sample_count == 1:
        axes = np.expand_dims(axes, axis=0)
    for index in range(sample_count):
        mask_array = masks[index].cpu().numpy().astype(np.float32)
        mask_array[mask_array == 255] = 0.5
        axes[index, 0].imshow(denormalize_image(images[index]))
        axes[index, 0].set_title(f"Sample {index + 1}: original image")
        axes[index, 0].axis("off")
        axes[index, 1].imshow(mask_array, cmap="gray", vmin=0, vmax=1)
        axes[index, 1].set_title(f"Sample {index + 1}: target mask")
        axes[index, 1].axis("off")
    plt.tight_layout()
    plt.show()


def plot_training_histories(fcn_history: Dict[str, List[float]], unet_history: Dict[str, List[float]]) -> None:
    figure, (axis_loss, axis_miou) = plt.subplots(1, 2, figsize=(15, 5))
    axis_loss.plot(fcn_history["val_loss"], label="FCN validation loss", color="red", linestyle="--")
    axis_loss.plot(unet_history["val_loss"], label="UNet validation loss", color="green", linewidth=2)
    axis_loss.set_title("FCN vs UNet: validation loss")
    axis_loss.grid(True)
    axis_loss.legend()
    axis_miou.plot(fcn_history["val_miou"], label="FCN validation mIoU", color="red", linestyle="--")
    axis_miou.plot(unet_history["val_miou"], label="UNet validation mIoU", color="green", linewidth=2)
    axis_miou.set_title("FCN vs UNet: validation mIoU")
    axis_miou.grid(True)
    axis_miou.legend()
    plt.tight_layout()
    plt.show()


def compare_fcn_unet_predictions(test_loader: DataLoader, device: torch.device) -> None:
    fcn_model = load_checkpoint(SimpleFCN(n_class=2).to(device), BEST_FCN_MODEL_PATH, device)
    unet_model = load_checkpoint(UNet(n_class=2).to(device), BEST_UNET_MODEL_PATH, device)
    fcn_model.eval()
    unet_model.eval()
    images, masks = next(iter(test_loader))
    images = images.to(device)
    with torch.no_grad():
        fcn_predictions = torch.argmax(fcn_model(images), dim=1).cpu().numpy()
        unet_predictions = torch.argmax(unet_model(images), dim=1).cpu().numpy()
    images_np = images.cpu().permute(0, 2, 3, 1).numpy()
    masks_np = masks.cpu().numpy()
    sample_count = min(16, len(images_np))
    figure, axes = plt.subplots(sample_count, 4, figsize=(16, 4 * sample_count))
    if sample_count == 1:
        axes = np.expand_dims(axes, axis=0)
    for index in range(sample_count):
        axes[index, 0].imshow(images_np[index]); axes[index, 0].axis("off")
        axes[index, 1].imshow(np.ma.masked_where(masks_np[index] == 255, masks_np[index]), cmap="gray"); axes[index, 1].axis("off")
        axes[index, 2].imshow(fcn_predictions[index], cmap="gray"); axes[index, 2].axis("off")
        axes[index, 3].imshow(unet_predictions[index], cmap="gray"); axes[index, 3].axis("off")
    axes[0, 0].set_title("Original image")
    axes[0, 1].set_title("Ground-truth")
    axes[0, 2].set_title("FCN prediction")
    axes[0, 3].set_title("UNet prediction")
    plt.tight_layout()
    plt.show()


def find_corrupted_data(dataloader: DataLoader, dataset_name: str, device: torch.device) -> None:
    bad_count = 0
    total_count = 0
    with torch.no_grad():
        for _, masks in dataloader:
            masks = masks.to(device)
            total_count += masks.size(0)
            bad_count += ((masks == 1).sum(dim=(1, 2)) == 0).sum().item()
    print(f"--- {dataset_name} data health report ---")
    print(f"Total images inspected: {total_count}")
    print(f"Empty masks detected: {bad_count}")
    if total_count > 0:
        print(f"Invalid mask ratio: {(bad_count / total_count) * 100:.2f}%")
    print("-" * 45)


def plot_sample_grid(samples: List[Dict[str, torch.Tensor]], title: str) -> None:
    if not samples:
        print(f"No qualifying samples were found for '{title}'.")
        return
    figure, axes = plt.subplots(len(samples), 4, figsize=(16, 4 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)
    figure.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    for row, sample in enumerate(samples):
        axes[row, 0].imshow(denormalize_image(sample["image"])); axes[row, 0].set_title("Original image"); axes[row, 0].axis("off")
        axes[row, 1].imshow(sample["mask"], cmap="gray"); axes[row, 1].set_title("Ground-truth"); axes[row, 1].axis("off")
        axes[row, 2].imshow(sample["baseline_prediction"], cmap="gray"); axes[row, 2].set_title(f"Baseline IoU: {sample['baseline_iou'] * 100:.1f}%"); axes[row, 2].axis("off")
        axes[row, 3].imshow(sample["augmented_prediction"], cmap="gray"); axes[row, 3].set_title(f"Augmented IoU: {sample['augmented_iou'] * 100:.1f}%"); axes[row, 3].axis("off")
    plt.tight_layout()
    plt.show()


def compare_augmentation_effects(test_loader: DataLoader, device: torch.device) -> None:
    baseline_model = load_checkpoint(UNet(n_class=2).to(device), BEST_UNET_MODEL_PATH, device)
    augmented_model = load_checkpoint(UNet(n_class=2).to(device), BEST_AUG_UNET_MODEL_PATH, device)
    baseline_model.eval(); augmented_model.eval()
    improved_samples: List[Dict[str, torch.Tensor]] = []
    failed_samples: List[Dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            baseline_predictions = torch.argmax(baseline_model(images), dim=1)
            augmented_predictions = torch.argmax(augmented_model(images), dim=1)
            for index in range(images.size(0)):
                baseline_iou = calculate_iou_for_image(baseline_predictions[index], masks[index])
                augmented_iou = calculate_iou_for_image(augmented_predictions[index], masks[index])
                sample = {"image": images[index].cpu(), "mask": masks[index].cpu(), "baseline_prediction": baseline_predictions[index].cpu(), "augmented_prediction": augmented_predictions[index].cpu(), "baseline_iou": baseline_iou, "augmented_iou": augmented_iou}
                if augmented_iou > baseline_iou + 0.15 and len(improved_samples) < 5:
                    improved_samples.append(sample)
                if augmented_iou < 0.40 and baseline_iou < 0.40 and len(failed_samples) < 5:
                    failed_samples.append(sample)
            if len(improved_samples) >= 5 and len(failed_samples) >= 5:
                break
    plot_sample_grid(improved_samples, "Examples improved by augmentation")
    plot_sample_grid(failed_samples, "Persistent hard examples")


def compare_transfer_predictions(test_loader: DataLoader, device: torch.device) -> None:
    scratch_model = load_checkpoint(UNet(n_class=2).to(device), BEST_UNET_MODEL_PATH, device)
    transfer_model = load_checkpoint(ResNet18UNet(pretrained=False, n_class=2).to(device), BEST_TRANSFER_MODEL_PATH, device)
    scratch_model.eval(); transfer_model.eval()
    images, masks = next(iter(test_loader))
    sample_count = min(4, images.size(0))
    figure, axes = plt.subplots(sample_count, 4, figsize=(16, 4 * sample_count))
    if sample_count == 1:
        axes = np.expand_dims(axes, axis=0)
    with torch.no_grad():
        for index in range(sample_count):
            image = images[index]
            mask = masks[index]
            image_input = image.unsqueeze(0).to(device)
            scratch_prediction = torch.argmax(scratch_model(image_input), dim=1).squeeze(0).cpu()
            transfer_prediction = torch.argmax(transfer_model(image_input), dim=1).squeeze(0).cpu()
            axes[index, 0].imshow(denormalize_image(image)); axes[index, 0].set_title(f"Sample {index + 1}: original"); axes[index, 0].axis("off")
            axes[index, 1].imshow(mask.cpu(), cmap="gray"); axes[index, 1].set_title("Ground-truth"); axes[index, 1].axis("off")
            axes[index, 2].imshow(scratch_prediction, cmap="gray"); axes[index, 2].set_title("Scratch UNet"); axes[index, 2].axis("off")
            axes[index, 3].imshow(transfer_prediction, cmap="gray"); axes[index, 3].set_title("ResNet18 UNet"); axes[index, 3].axis("off")
    plt.tight_layout(); plt.show()


def compare_semantic_vs_instance(test_loader: DataLoader, device: torch.device) -> None:
    require_dependency(torchvision, "torchvision", "Mask R-CNN comparison")
    require_dependency(ndimage, "scipy", "instance counting")
    semantic_model = load_checkpoint(ResNet18UNet(pretrained=False, n_class=2).to(device), BEST_TRANSFER_MODEL_PATH, device)
    instance_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    semantic_model.eval(); instance_model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    images, _ = next(iter(test_loader))
    images = images.to(device)
    sample_count = min(10, images.size(0))
    figure, axes = plt.subplots(sample_count, 3, figsize=(15, 4 * sample_count))
    if sample_count == 1:
        axes = np.expand_dims(axes, axis=0)
    with torch.no_grad():
        semantic_outputs = semantic_model(images)
        semantic_predictions = torch.argmax(semantic_outputs, dim=1)
        for index in range(sample_count):
            image_cpu = images[index].cpu().clone()
            for channel, mean, std in zip(image_cpu, DEFAULT_MEAN, DEFAULT_STD):
                channel.mul_(std).add_(mean)
            single_image = torch.clamp(image_cpu, 0, 1).to(device)
            instance_output = instance_model([single_image])[0]
            image_display = single_image.cpu().permute(1, 2, 0).numpy()
            axes[index, 0].imshow(image_display); axes[index, 0].set_title("Original image"); axes[index, 0].axis("off")
            semantic_mask = semantic_predictions[index].cpu().numpy()
            labeled_mask, instance_count = ndimage.label((semantic_mask == 1).astype(np.uint8))
            axes[index, 1].imshow(image_display); axes[index, 1].imshow(labeled_mask, cmap="jet", alpha=0.5); axes[index, 1].set_title(f"Semantic U-Net ({instance_count})"); axes[index, 1].axis("off")
            for object_index in range(1, instance_count + 1):
                positions = np.where(labeled_mask == object_index)
                if len(positions[0]) > 0:
                    y_min, y_max = np.min(positions[0]), np.max(positions[0])
                    x_min, x_max = np.min(positions[1]), np.max(positions[1])
                    axes[index, 1].add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="red", facecolor="none"))
            axes[index, 2].imshow(image_display); axes[index, 2].axis("off")
            valid_indices = np.where(instance_output["scores"].cpu().numpy() > 0.5)[0]
            axes[index, 2].set_title(f"Mask R-CNN ({len(valid_indices)})")
            for valid_index in valid_indices:
                x_min, y_min, x_max, y_max = instance_output["boxes"][valid_index].cpu().numpy()
                axes[index, 2].add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="lime", facecolor="none"))
                instance_mask = instance_output["masks"][valid_index, 0].cpu().numpy()
                axes[index, 2].imshow(np.ma.masked_where(instance_mask < 0.5, instance_mask), cmap="Greens", alpha=0.5)
    plt.tight_layout(); plt.show()


def analyze_image_size_distribution(paths: DataPaths) -> None:
    widths: List[int] = []
    heights: List[int] = []
    for image_path in paths.image_dir.glob("*.jpg"):
        with Image.open(image_path) as image:
            width, height = image.size
            widths.append(width)
            heights.append(height)
    print("--- Original image size distribution ---")
    print(f"Width  -> min: {np.min(widths)} px | max: {np.max(widths)} px | mean: {np.mean(widths):.0f} px")
    print(f"Height -> min: {np.min(heights)} px | max: {np.max(heights)} px | mean: {np.mean(heights):.0f} px")


def compare_yolo_runtime(test_loader: DataLoader, device: torch.device) -> None:
    require_dependency(YOLO, "ultralytics", "YOLO runtime comparison")
    unet_model = load_checkpoint(UNet(n_class=2).to(device), BEST_UNET_MODEL_PATH, device)
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    unet_model.eval()
    images, _ = next(iter(test_loader))
    sample_count = min(5, len(images))
    figure, axes = plt.subplots(sample_count, 3, figsize=(18, 5 * sample_count))
    if sample_count == 1:
        axes = np.expand_dims(axes, axis=0)
    total_unet_time = 0.0
    total_yolo_time = 0.0
    for index in range(sample_count):
        image_tensor = images[index]
        image_np = denormalize_image(image_tensor)
        image_yolo = np.ascontiguousarray((image_np * 255).astype(np.uint8))
        yolo_start = time.perf_counter(); yolo_result = yolo_model(image_yolo, verbose=False); yolo_duration = time.perf_counter() - yolo_start
        total_yolo_time += yolo_duration
        unet_start = time.perf_counter()
        with torch.no_grad():
            unet_output = unet_model(image_tensor.unsqueeze(0).to(device))
            unet_prediction = torch.argmax(unet_output, dim=1).squeeze(0).cpu()
        unet_duration = time.perf_counter() - unet_start
        total_unet_time += unet_duration
        axes[index, 0].imshow(image_np); axes[index, 0].set_title("Original image"); axes[index, 0].axis("off")
        axes[index, 1].imshow(unet_prediction, cmap="gray"); axes[index, 1].set_title(f"U-Net ({unet_duration:.4f} s)"); axes[index, 1].axis("off")
        axes[index, 2].imshow(yolo_result[0].plot()[..., ::-1]); axes[index, 2].set_title(f"YOLOv8 ({yolo_duration:.4f} s)"); axes[index, 2].axis("off")
    plt.tight_layout(); plt.show()
    print(f"Average U-Net inference time: {total_unet_time / sample_count:.4f} seconds/image")
    print(f"Average YOLOv8 inference time: {total_yolo_time / sample_count:.4f} seconds/image")


def print_parameter_summary(device: torch.device) -> None:
    print("--- Model parameter counts ---")
    print(f"SimpleFCN: {count_model_parameters(SimpleFCN(n_class=2).to(device))}")
    print(f"UNet: {count_model_parameters(UNet(n_class=2).to(device))}")
    if smp is not None:
        print(f"ResNet18 U-Net: {count_model_parameters(ResNet18UNet(pretrained=False, n_class=2).to(device))}")


def validate_data_paths(paths: DataPaths) -> None:
    required_paths = [paths.image_dir, paths.annotations_dir, paths.trimap_dir, paths.trainval_path, paths.test_path]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_block = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"The following required dataset paths are missing:\n{missing_block}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic segmentation pipeline for the Oxford-IIIT Pet dataset")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-sample", action="store_true")
    parser.add_argument("--preview-loader", action="store_true")
    parser.add_argument("--analyze-class-balance", action="store_true")
    parser.add_argument("--analyze-trimap", action="store_true")
    parser.add_argument("--smoke-test-models", action="store_true")
    parser.add_argument("--train-fcn", action="store_true")
    parser.add_argument("--train-unet", action="store_true")
    parser.add_argument("--train-augmented-unet", action="store_true")
    parser.add_argument("--train-transfer", action="store_true")
    parser.add_argument("--plot-histories", action="store_true")
    parser.add_argument("--check-data-health", action="store_true")
    parser.add_argument("--compare-fcn-unet", action="store_true")
    parser.add_argument("--compare-augmentation", action="store_true")
    parser.add_argument("--compare-transfer", action="store_true")
    parser.add_argument("--compare-semantic-instance", action="store_true")
    parser.add_argument("--analyze-image-sizes", action="store_true")
    parser.add_argument("--compare-yolo", action="store_true")
    parser.add_argument("--count-parameters", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    selected_actions = [args.preview_sample, args.preview_loader, args.analyze_class_balance, args.analyze_trimap, args.smoke_test_models, args.train_fcn, args.train_unet, args.train_augmented_unet, args.train_transfer, args.plot_histories, args.check_data_health, args.compare_fcn_unet, args.compare_augmentation, args.compare_transfer, args.compare_semantic_instance, args.analyze_image_sizes, args.compare_yolo, args.count_parameters]
    if not any(selected_actions):
        parser.print_help()
        return
    config = RuntimeConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    paths = DataPaths(data_root=args.data_root)
    device = get_device(args.device)
    print(f"Active device: {device}")
    if args.smoke_test_models:
        smoke_test_models(device, config.image_size)
    if args.count_parameters:
        print_parameter_summary(device)
    if args.analyze_image_sizes:
        validate_data_paths(paths)
        analyze_image_size_distribution(paths)
    needs_standard_data = any([args.preview_sample, args.preview_loader, args.analyze_class_balance, args.analyze_trimap, args.train_fcn, args.train_unet, args.train_transfer, args.plot_histories, args.check_data_health, args.compare_fcn_unet, args.compare_augmentation, args.compare_transfer, args.compare_semantic_instance, args.compare_yolo])
    train_loader = test_loader = None
    fcn_history = unet_history = None
    if needs_standard_data:
        validate_data_paths(paths)
        train_loader, test_loader, train_list, test_list = create_standard_loaders(paths, config)
        print(f"Found {len(train_list)} training images and {len(test_list)} validation images.")
    if args.preview_sample:
        preview_mask_conversion(paths)
    if args.preview_loader and train_loader is not None:
        preview_dataloader_batch(train_loader)
    if args.analyze_class_balance and train_loader is not None:
        analyze_class_balance(train_loader)
    if args.analyze_trimap and train_loader is not None:
        analyze_trimap_statistics(train_loader)
    if args.train_fcn and train_loader is not None and test_loader is not None:
        fcn_model = SimpleFCN(n_class=2).to(device)
        fcn_history, best_miou = train_model(fcn_model, train_loader, test_loader, optim.Adam(fcn_model.parameters(), lr=config.learning_rate), nn.CrossEntropyLoss(ignore_index=255), device, config.num_epochs, BEST_FCN_MODEL_PATH, "FCN Train", "FCN Validation")
        print(f"SimpleFCN training complete. Best mIoU: {best_miou * 100:.2f}%")
    if args.train_unet and train_loader is not None and test_loader is not None:
        unet_model = UNet(n_class=2).to(device)
        unet_history, best_miou = train_model(unet_model, train_loader, test_loader, optim.Adam(unet_model.parameters(), lr=config.learning_rate), nn.CrossEntropyLoss(ignore_index=255), device, config.num_epochs, BEST_UNET_MODEL_PATH, "UNet Train", "UNet Validation")
        print(f"UNet training complete. Best mIoU: {best_miou * 100:.2f}%")
    if args.train_augmented_unet:
        validate_data_paths(paths)
        augmented_train_loader, augmented_test_loader = create_augmented_loaders(paths, config)
        augmented_model = UNet(n_class=2).to(device)
        _, best_miou = train_model(augmented_model, augmented_train_loader, augmented_test_loader, optim.Adam(augmented_model.parameters(), lr=config.learning_rate), nn.CrossEntropyLoss(ignore_index=255), device, config.num_epochs, BEST_AUG_UNET_MODEL_PATH, "Augmented UNet Train", "Augmented UNet Validation")
        print(f"Augmented UNet training complete. Best mIoU: {best_miou * 100:.2f}%")
    if args.train_transfer and train_loader is not None and test_loader is not None:
        _, transfer_history = train_transfer_model(train_loader, test_loader, config, device)
        print(f"Transfer learning complete. Collected {len(transfer_history['val_miou'])} validation points.")
    if args.plot_histories:
        if fcn_history is None or unet_history is None:
            print("History plotting requires --train-fcn and --train-unet in the same session.")
        else:
            plot_training_histories(fcn_history, unet_history)
    if args.check_data_health and train_loader is not None and test_loader is not None:
        find_corrupted_data(train_loader, "Train", device)
        find_corrupted_data(test_loader, "Validation", device)
    if args.compare_fcn_unet and test_loader is not None:
        compare_fcn_unet_predictions(test_loader, device)
    if args.compare_augmentation and test_loader is not None:
        compare_augmentation_effects(test_loader, device)
    if args.compare_transfer and test_loader is not None:
        compare_transfer_predictions(test_loader, device)
    if args.compare_semantic_instance and test_loader is not None:
        compare_semantic_vs_instance(test_loader, device)
    if args.compare_yolo and test_loader is not None:
        compare_yolo_runtime(test_loader, device)


if __name__ == "__main__":
    main()

