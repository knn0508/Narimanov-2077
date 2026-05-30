FROM python:3.11-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies using CPU-only index to avoid downloading 2GB+ of CUDA libraries
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 7860 (default port for Hugging Face Spaces)
ENV PORT=7860
EXPOSE 7860

# Start uvicorn server
CMD ["sh", "-c", "uvicorn smartwave_ai.app:app --host 0.0.0.0 --port $PORT"]
