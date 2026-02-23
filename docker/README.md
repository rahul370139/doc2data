# Docker Configuration

Docker-related files for Doc2Data.

## Contents

- **Dockerfile** – Image definition (NVIDIA PyTorch base, Detectron2, PaddleOCR, etc.)
- **requirements_docker.txt** – Python dependencies for the container
- **start_services.sh** – Container entrypoint (starts FastAPI + Streamlit)
- **docker-compose.yml** – Compose file for API + Streamlit services (use from project root)

## Usage

### From project root

```bash
# Build and run (compose file is in docker/)
docker-compose -f docker/docker-compose.yml up --build

# Or build only
docker build -f docker/Dockerfile -t doc2data-gpu .
```

### GPU run

```bash
docker run -d --gpus all --name doc2data-gpu-app \
  -p 8501:8501 -p 8000:8000 -p 11434:11434 \
  -v "$(pwd)/data:/app/data" \
  -e USE_GPU=true \
  -e ENABLE_SLM=true \
  -e ENABLE_VLM=true \
  doc2data-gpu
```

**Access:** http://localhost:8501 (Streamlit)
