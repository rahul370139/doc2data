# Doc2Data API Deployment Guide

## Quick Start (DGX)

### 1. Start the API Server

```bash
# Navigate to project root
cd /path/to/doc2data

# Option A: Run in foreground (for testing)
./scripts/start_api_server.sh --port 8000 --workers 4

# Option B: Run in background with nohup
mkdir -p logs
nohup ./scripts/start_api_server.sh --port 8000 --workers 4 > logs/api.log 2>&1 &

# Check if running
curl http://localhost:8000/health
```

### 2. Share with Others

The API is accessible at:
```
http://<DGX_IP>:8000
```

Get your DGX IP:
```bash
hostname -I | awk '{print $1}'
```

Share these endpoints with your team:
- **API Docs**: `http://<DGX_IP>:8000/docs` (Interactive Swagger UI)
- **Health Check**: `http://<DGX_IP>:8000/health`
- **Extract V2**: `http://<DGX_IP>:8000/extract/v2`
- **CMS-1500**: `http://<DGX_IP>:8000/extract/cms1500`
- **UB-04**: `http://<DGX_IP>:8000/extract/ub04`
- **Reducto Format**: `http://<DGX_IP>:8000/extract/reducto`

## API Usage Examples

### Python Client

```python
import requests

# API base URL (replace with your DGX IP)
BASE_URL = "http://<DGX_IP>:8000"

# Extract from any form (auto-detect)
def extract_document(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/extract/v2",
            files={"file": f}
        )
    return response.json()

# Extract UB-04 specifically
def extract_ub04(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/extract/ub04",
            files={"file": f}
        )
    return response.json()

# Example usage
result = extract_document("path/to/ub04_form.pdf")
print(f"Form Type: {result['form_type']}")
print(f"Extracted Fields: {len(result['extracted_fields'])}")
print(f"Business Fields: {result.get('business_fields', {})}")
```

### cURL

```bash
# Health check
curl http://<DGX_IP>:8000/health

# Extract document
curl -X POST "http://<DGX_IP>:8000/extract/v2" \
  -F "file=@/path/to/document.pdf"

# Extract UB-04
curl -X POST "http://<DGX_IP>:8000/extract/ub04" \
  -F "file=@/path/to/ub04_form.pdf"
```

### JavaScript/Fetch

```javascript
const extractDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://<DGX_IP>:8000/extract/v2', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

## Production Deployment (Systemd)

For running as a system service:

```bash
# 1. Copy and edit the service file
sudo cp deployment/doc2data-api.service /etc/systemd/system/
sudo nano /etc/systemd/system/doc2data-api.service
# Update: User, WorkingDirectory, ExecStart paths

# 2. Reload systemd and start
sudo systemctl daemon-reload
sudo systemctl enable doc2data-api
sudo systemctl start doc2data-api

# 3. Check status
sudo systemctl status doc2data-api

# 4. View logs
journalctl -u doc2data-api -f
```

## Docker Deployment (Alternative)

```bash
# Build
docker build -t doc2data-api .

# Run
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --name doc2data-api \
  doc2data-api
```

## Troubleshooting

### Check if server is running
```bash
ps aux | grep uvicorn
```

### Kill stuck process
```bash
pkill -f "uvicorn app.api_main"
```

### View logs
```bash
tail -f logs/api.log
```

### Common Issues

1. **Port already in use**: 
   ```bash
   lsof -i :8000
   kill <PID>
   ```

2. **ModuleNotFoundError**: Activate virtual environment first
   ```bash
   source venv/bin/activate
   ```

3. **CORS errors**: API allows all origins by default. For stricter settings, edit `api_main.py`.
