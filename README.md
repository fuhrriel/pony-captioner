# Pony Captioner - Docker Edition

A standalone Python script to generate detailed, V7-compatible captions for images using multiple ML models. This version is optimized for Docker deployment with GPU support.

**GitHub Repository:** https://github.com/fuhrriel/pony-captioner

## What's Included

- `pony_captioner.py` - Main captioning script
- `test_gpu.py` - GPU/CUDA verification script  
- `Dockerfile` - Container definition with CUDA 12.1 support
- `docker-compose.yml` - Easy orchestration setup
- `requirements.txt` - Python dependencies reference
- `.env.example` - Environment variable template
- `example_usage.sh` - Example commands
- `README.md` - This file

## Features

- **Multi-model tagging** using SmilingWolf/wd-swinv2-tagger-v3 and toynya/Z3D-E621-Convnext
- **Content captioning** with detailed descriptions of subjects, actions, and environment
- **Style captioning** analyzing artistic elements like composition, lighting, and medium
- **Style clustering** grouping images into 2048 style categories
- **Aesthetic scoring** rating images from 0-9 based on quality
- **Automatic caption assembly** combining all elements into training-ready captions

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA 12.6+ support and at least 40GB VRAM (for the largest models)
- ~100GB disk space for models and cache

**Note:** The Docker image uses Ubuntu 24.04, Python 3.12, CUDA 12.6, and installs PyTorch 2.8.0 with CUDA support from PyTorch's official wheel repository.

### Running Without Docker (Advanced)

If you want to run without Docker on your host system:

```bash
# Clone the repository
git clone https://github.com/fuhrriel/pony-captioner.git
cd pony-captioner

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install numpy==2.0.2 transformers==4.44.2 lmdeploy==0.10.1 rich pandas \
    onnxruntime-gpu git+https://github.com/openai/CLIP.git huggingface-hub Pillow

# Run the script
python pony_captioner.py /path/to/images
```

**Requirements for non-Docker:**
- CUDA 12.6+ drivers installed
- Python 3.10+ (3.12 recommended)
- ~100GB for models

## Setup

### 1. Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Prepare Directory Structure

```bash
mkdir -p pony-captioner/images pony-captioner/models
cd pony-captioner
```

### 3. Add Files

Place these files in the `pony-captioner` directory:
- `pony_captioner.py`
- `test_gpu.py`
- `Dockerfile`
- `compose.yml`

Optional: Create `.env` file from template for custom configuration:
```bash
cp .env.example .env
# Edit .env with your settings (HuggingFace token, etc.)
```

### 4. Add Your Images

Copy your images to the `images` folder:

```bash
cp /path/to/your/images/* ./images/
```

## Usage

### Build the Docker Image

```bash
docker compose build
```

### Test GPU Support (Recommended First Step)

Before processing images, verify that CUDA and GPU are working correctly:

```bash
docker compose run --rm --entrypoint python3 pony-captioner /workspace/test_gpu.py
```

This should show:
- ✓ CUDA available: True
- ✓ GPU information
- ✓ All dependencies loaded

### Run the Captioner

Basic usage (process all images in the images folder):

```bash
docker compose run --rm pony-captioner /workspace/images
```

Force regeneration of all captions:

```bash
docker compose run --rm pony-captioner /workspace/images --force-regen
```

With verbose output:

```bash
docker compose run --rm pony-captioner /workspace/images --verbose
```

Use manual style cluster (for training new styles):

```bash
docker compose run --rm pony-captioner /workspace/images --style-cluster 2048
```

This skips automatic style clustering and uses your provided cluster ID instead.

### Using Without Docker Compose

```bash
docker build -t pony-captioner .

docker run --rm --gpus all \
  -v $(pwd)/images:/workspace/images \
  -v $(pwd)/models:/models \
  --shm-size=16g \
  pony-captioner /workspace/images
```

## Output Files

For each image (e.g., `example.png`), the script generates:

- `example.tags.json` - Tags and rating from both taggers
- `example.caption.txt` - Detailed content description
- `example.style_caption.txt` - Style and artistic analysis
- `example.cluster.txt` - Style cluster ID
- `example.score.txt` - Aesthetic score
- `example.txt` - **Final combined caption for training**

## Models Downloaded

On first run, the following models will be downloaded to `./models`:

1. **SmilingWolf/wd-swinv2-tagger-v3** (~1GB) - Image tagging
2. **toynya/Z3D-E621-Convnext** (~500MB) - Additional tagging
3. **purplesmartai/Pony-InternVL2-40B-AWQ** (~20GB) - Content captioning
4. **purplesmartai/Pony-InternVL2-26B-AWQ** (~13GB) - Style captioning
5. **purplesmartai/style-classifier** (~2GB) - Style clustering
6. **purplesmartai/aesthetic-classifier** (~300MB) - Quality scoring
7. **CLIP ViT-L/14** (~1.7GB) - Used by clustering and aesthetic models

Total: ~40GB

## Command Line Options

```
usage: pony_captioner.py [-h] [--force-regen] [--verbose] [--style-cluster STYLE_CLUSTER] image_folder

positional arguments:
  image_folder          Path to folder containing images

options:
  -h, --help            show this help message and exit
  --force-regen         Force regeneration of all files even if they exist
  --verbose             Enable verbose output
  --style-cluster STYLE_CLUSTER
                        Manual style cluster ID (skips automatic clustering)
```

### Examples

```bash
# Basic usage
python pony_captioner.py /path/to/images

# Force regenerate everything
python pony_captioner.py /path/to/images --force-regen

# Use manual style cluster (useful for training new styles)
python pony_captioner.py /path/to/images --style-cluster 2048

# Combine options
python pony_captioner.py /path/to/images --style-cluster 2048 --verbose
```

## Performance Notes

- First run will take longer due to model downloads
- Processing time varies by image size and GPU performance
- The 40B content model requires ~40GB VRAM
- Smaller GPUs may need to use smaller models or reduce batch sizes

## Troubleshooting

### Verify GPU and CUDA Support

Check if PyTorch can see your GPU:

```bash
docker compose run --rm --entrypoint python3 pony-captioner -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output should show `CUDA available: True`.

### Out of Memory Errors

If you encounter OOM errors:
1. Ensure you have enough VRAM (40GB+ recommended)
2. Close other GPU-intensive applications
3. Process images in smaller batches
4. Increase `shm_size` in compose.yml

### Models Not Downloading

- Check your internet connection
- Verify you have enough disk space (~100GB free)
- Check HuggingFace is accessible from your network

### Permission Errors

```bash
sudo chown -R $USER:$USER ./images ./models
```

## Advanced Configuration

### Using a Different Image Folder

Edit `compose.yml` and change the volume mount:

```yaml
volumes:
  - /path/to/your/images:/workspace/images
  - ./models:/models
```

### Persistent Model Cache

Models are cached in the `./models` directory and persist between runs. To clear the cache:

```bash
rm -rf ./models/huggingface_cache/*
```

### Running Specific Steps Only

Modify the script to skip certain steps by commenting out the relevant processing in the `process_image()` method.

## License

This tool uses multiple models with different licenses. Please review each model's license:
- Check HuggingFace model cards for license information
- Ensure compliance with all model licenses for your use case

## Credits

Based on the Pony Captioner Colab notebook by the Pony V7 team.
