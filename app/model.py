"""
Semantic Segmentation Model Interface.

This module handles the initialization, weight loading, and inference 
pipeline for the ResNet18 U-Net pet segmentation model.
"""

import io

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torchvision import transforms


class InvalidImage(ValueError):
    """Uploaded bytes could not be decoded as an image."""


class PetSegmentationModel:
    """
    A class to wrap the PyTorch segmentation model for API serving.
    Handles device management, preprocessing, and postprocessing.
    """

    def __init__(self, weight_path: str) -> None:
        """
        Initializes the model and preprocessing pipeline.

        Args:
            weight_path (str): The file path to the trained model weights (.pth).
        """
        # Use GPU if available, otherwise default to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(weight_path)
        
        # Preprocessing pipeline
        # Note: Normalization (mean/std) is intentionally omitted to match
        # the [0.0, 1.0] scaling used during the training phase.
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _load_model(self, weight_path: str) -> torch.nn.Module:
        """
        Constructs the model architecture and loads the pre-trained weights.

        Args:
            weight_path (str): The file path to the trained model weights.

        Returns:
            torch.nn.Module: The loaded and configured PyTorch model.
        """
        # Initialize architecture: ResNet18 encoder, U-Net decoder, 2 classes
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,  # We are loading fine-tuned weights
            in_channels=3,
            classes=2,
        )
        
        # Load weights securely
        state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        
        # Move model to device and set to evaluation mode
        model.to(self.device)
        model.eval()
        
        return model

    def predict(self, image_bytes: bytes) -> Image.Image:
        """
        Processes raw image bytes and returns a segmentation mask.

        Args:
            image_bytes (bytes): The raw image data received from the API.

        Returns:
            Image.Image: A PIL Image representing the binary mask 
                         (0: Background, 255: Pet).
        """
        # 1. Preprocessing
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
        except (OSError, SyntaxError, ValueError) as error:
            raise InvalidImage("Could not decode the uploaded image.") from error
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # 2. Inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # 3. Postprocessing
        # Output shape: [1, 2, 224, 224] -> Argmax to select class 0 or 1
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Scale to 0 and 255 for visual/HTTP response compatibility
        mask_image_array = (mask * 255).astype(np.uint8)
        
        return Image.fromarray(mask_image_array, mode="L")
