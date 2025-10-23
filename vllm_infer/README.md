# Hunyuan Image3 API Service

Fast text-to-image generation with vLLM backend.

## Quick Start

### 1. Launch Server
```bash
./run_vllm_server.sh /path/to/model
```
Server starts at `http://localhost:8000`

### 2. Generate Images
```bash
python openai_client.py --prompt "your image description"
```

## Prerequisites

### Install Dependencies
```bash
# Install hunyuan_image_3 package first
pip install hunyuan_image_3

# Install vLLM from specific commit
# https://github.com/kippergong/vllm/tree/feature/hunyuan_image_3.0
```

### Verified Environment
The following environment has been tested and verified to work:
- **PyTorch**: 2.7.1
- **CUDA**: 12.8
- **vLLM**: https://github.com/kippergong/vllm/tree/feature/hunyuan_image_3.0
- **hunyuan_image_3**: Latest version

## Examples

**Basic:**
```bash
python openai_client.py --prompt "panda eating bamboo in forest"
```

**Custom size:**
```bash
python openai_client.py --width 768 --height 512 --prompt "sunset beach"
```

**Auto shape:**
```bash
python openai_client.py --bot-task "auto" --prompt "sunset beach"
```

## Common Options
- `--prompt`: Text description (required)
- `--width/--height`: Image size (default: 1024x1024)
- `--seed`: Reproducible results
- `--bot-task`: Task type (image/auto/recaption). Note: recaption requires an 'instruct' checkpoint and is not yet supported.