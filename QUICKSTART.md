# Quick Start Guide

**GitHub:** https://github.com/fuhrriel/pony-captioner

## Prerequisites

- NVIDIA GPU with 40GB+ VRAM
- Docker with NVIDIA Container Toolkit installed
- ~100GB free disk space

## Setup (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/fuhrriel/pony-captioner.git
cd pony-captioner

# 2. Create directories for images and models
mkdir -p images models

# 3. Copy your images
cp /path/to/your/images/* ./images/

# 4. Build the container
docker compose build
```

## Verify GPU Works

```bash
docker compose run --rm --entrypoint python3 pony-captioner /workspace/test_gpu.py
```

**Expected output:**
```
✓ Python version: 3.12.x
✓ PyTorch version: 2.8.0+cu126
✓ CUDA available: True
✓ CUDA version: 12.6
✓ GPU count: 1
  - GPU 0: NVIDIA A100-SXM4-40GB
    Memory: 40.0 GB
```

## Run the Captioner

```bash
# Process all images in ./images folder
docker compose run --rm pony-captioner /workspace/images
```

First run will download ~40GB of models. Be patient!

## Check Results

```bash
# View a generated caption
cat images/your_image.txt
```

## Common Commands

```bash
# Force regenerate everything
docker compose run --rm pony-captioner /workspace/images --force-regen

# Use manual style cluster (for training new styles not in original dataset)
docker compose run --rm pony-captioner /workspace/images --style-cluster 2048

# Verbose output for debugging
docker compose run --rm pony-captioner /workspace/images --verbose

# Open shell inside container
docker compose run --rm --entrypoint /bin/bash pony-captioner

# Check GPU from inside container
docker compose run --rm --entrypoint nvidia-smi pony-captioner
```

## File Outputs

For each `image.png`, you'll get:
- `image.tags.json` - Extracted tags and rating
- `image.caption.txt` - Detailed content description
- `image.style_caption.txt` - Style analysis
- `image.cluster.txt` - Style cluster ID
- `image.score.txt` - Aesthetic score (0-9)
- **`image.txt`** - Final combined caption ← Use this for training!

## Troubleshooting

### "externally-managed-environment" Error During Build

If you see this error:
```
error: externally-managed-environment
× This environment is externally managed
```

**Solution**: Make sure your Dockerfile has this line at the top:
```dockerfile
ENV PIP_BREAK_SYSTEM_PACKAGES=1
```

This is normal for Ubuntu 24.04 and is safe in Docker. See `PEP668_UBUNTU24.md` for details.

### "CUDA not available"
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# If this fails, reinstall NVIDIA Container Toolkit
```

### "Out of memory"
- You need at least 40GB VRAM
- Close other GPU applications
- Try processing fewer images at once

### "Models won't download"
- Check internet connection
- Check disk space (need ~100GB free)
- Models cache in `./models/huggingface_cache/`

## Performance

- **First run**: ~2-5 minutes per image (model loading + download)
- **Subsequent runs**: ~30-60 seconds per image
- **With existing captions**: ~instant (only processes missing files)

## That's It!

The captioner will process all images and create detailed training-ready captions. Models are cached in `./models/` so subsequent runs are much faster.
