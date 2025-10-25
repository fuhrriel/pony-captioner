#!/bin/bash
# Example usage scripts for Pony Captioner

# Build the image
echo "Building Docker image..."
docker compose build

# Test GPU and CUDA support
echo "Testing GPU and CUDA support..."
docker compose run --rm --entrypoint python3 pony-captioner /workspace/test_gpu.py

# Example 1: Basic usage - process all images
echo "Example 1: Processing images..."
docker compose run --rm pony-captioner /workspace/images

# Example 2: Force regeneration of all captions
echo "Example 2: Force regeneration..."
docker compose run --rm pony-captioner /workspace/images --force-regen

# Example 3: Verbose output for debugging
echo "Example 3: Verbose mode..."
docker compose run --rm pony-captioner /workspace/images --verbose

# Example 4: Using docker directly without compose
echo "Example 4: Direct docker run..."
docker run --rm --gpus all \
  -v $(pwd)/images:/workspace/images \
  -v $(pwd)/models:/models \
  --shm-size=16g \
  pony-captioner /workspace/images

# Example 5: Process images from a different directory
echo "Example 5: Custom image directory..."
docker run --rm --gpus all \
  -v /path/to/your/images:/workspace/custom_images \
  -v $(pwd)/models:/models \
  --shm-size=16g \
  pony-captioner /workspace/custom_images

# Example 6: Interactive shell for debugging
echo "Example 6: Interactive shell..."
docker compose run --rm --entrypoint /bin/bash pony-captioner

# Check GPU availability
echo "Checking GPU availability..."
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi

# View generated captions
echo "Viewing sample caption..."
if [ -f images/*.full_caption.txt ]; then
  cat images/*.full_caption.txt | head -20
else
  echo "No captions generated yet. Run the captioner first."
fi
