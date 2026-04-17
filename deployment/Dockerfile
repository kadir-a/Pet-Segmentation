# Stage 1: Base Image (Using Debian Bookworm for stability and security)
FROM python:3.10-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Stage 2: Install system dependencies required by OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Stage 3: Copy requirements and install Python dependencies
# Done before copying the code to leverage Docker cache
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Stage 4: Copy application code and model weights
COPY app /app/app
COPY weights /app/weights

# Stage 5: Expose the API port and define the startup command
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]