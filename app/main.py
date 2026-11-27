"""
FastAPI application routing and endpoint definitions.

This module exposes the REST API endpoints for the Pet Semantic Segmentation service,
handling incoming image requests, validation, and serving the output masks.
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.model import InvalidImage, PetSegmentationModel


@asynccontextmanager
async def lifespan(app):
    weights = Path(__file__).resolve().parents[1] / "weights" / "best_transfer_model.pth"
    app.state.pet_model = PetSegmentationModel(str(weights))
    try:
        yield
    finally:
        del app.state.pet_model


# Initialize the FastAPI application with production metadata
app = FastAPI(
    title="Pet Semantic Segmentation API",
    description=("A production-grade REST API for generating semantic masks of cats and dogs "
                 "using a fine-tuned ResNet18 U-Net architecture."),
    version="2.0.0",
    lifespan=lifespan,
)

@app.get("/", tags=["Health"])
def health_check() -> dict:
    """
    Health check endpoint to verify the API is running.

    Returns:
        dict: A status dictionary.
    """
    return {
        "status": "active",
        "message": "API is running. Send a POST request to /predict to generate a mask."
    }

@app.post("/predict", tags=["Inference"])
async def predict_mask(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Accepts an uploaded image, processes it through the segmentation model,
    and returns the resulting binary mask.

    Args:
        file (UploadFile): The image file uploaded by the user.

    Returns:
        StreamingResponse: The predicted segmentation mask as a PNG image stream.
        
    Raises:
        HTTPException: If the uploaded file is not an image (400) or processing fails (500).
    """
    # 1. Security Check: Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload a valid image file (e.g., JPEG, PNG)."
        )

    try:
        # 2. Read the image data into memory
        image_bytes = await file.read()
        
        # 3. Perform inference using the loaded model
        mask_image = app.state.pet_model.predict(image_bytes)
        
        # 4. Convert the PIL Image to a byte stream for HTTP transmission
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)  # Reset stream pointer to the beginning
        
        # 5. Return the mask as an image response
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except InvalidImage as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        # Catch unexpected errors to prevent hard crashes and inform the client
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during processing: {str(e)}"
        )