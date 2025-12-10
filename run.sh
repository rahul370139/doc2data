#!/bin/bash
# Doc2Data - DGX Deployment Script
# Usage: ./run.sh [sync|run|deploy|local]
#
# Commands:
#   sync   - Sync files to DGX (rsync)
#   run    - Run Streamlit on DGX (assumes synced)
#   deploy - Sync + Run (full deployment)
#   local  - Run locally (for testing)

set -e

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# DGX2 Connection Details
DGX_HOST="100.126.216.92"
DGX_USER="radiant-dgx2"
SSH_KEY="/Users/rahul/Downloads/dgx-spark/tailscale_spark2"
REMOTE_DIR="/home/radiant-dgx2/doc2data"

# Ports
STREAMLIT_PORT=8501
API_PORT=8000

# Local project directory
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# ═══════════════════════════════════════════════════════════════════════════
# Colors & UI
# ═══════════════════════════════════════════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           ${BOLD}Doc2Data${NC}${PURPLE} - Intelligent Document Extraction        ║"
    echo "║              Healthcare Forms • GPU-Accelerated              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${CYAN}🔹 $1${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════
# SSH Helper
# ═══════════════════════════════════════════════════════════════════════════

ssh_cmd() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o BatchMode=yes "$DGX_USER@$DGX_HOST" "$@"
}

ssh_interactive() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$DGX_USER@$DGX_HOST" "$@"
}

# ═══════════════════════════════════════════════════════════════════════════
# Sync to DGX
# ═══════════════════════════════════════════════════════════════════════════

sync_to_dgx() {
    log_info "Syncing project to DGX2..."
    log_step "Source: $LOCAL_DIR"
    log_step "Target: $DGX_USER@$DGX_HOST:$REMOTE_DIR"
    
    # Create remote directory if needed
    ssh_cmd "mkdir -p $REMOTE_DIR"
    
    # Rsync with exclusions (smart sync - only changed files)
    rsync -avz --progress \
        --exclude='venv/' \
        --exclude='.venv/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.git/' \
        --exclude='cache/' \
        --exclude='*.egg-info/' \
        --exclude='.DS_Store' \
        --exclude='*.log' \
        --exclude='models/' \
        --exclude='.streamlit/' \
        -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$LOCAL_DIR/" "$DGX_USER@$DGX_HOST:$REMOTE_DIR/"
    
    log_success "Sync complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
# Setup DGX Environment
# ═══════════════════════════════════════════════════════════════════════════

setup_dgx_env() {
    log_info "Setting up DGX environment..."
    
    ssh_cmd << 'SETUP_EOF'
cd ~/doc2data

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install requirements
source venv/bin/activate

# Install/upgrade core dependencies
pip install -q --upgrade pip
pip install -q streamlit pandas numpy opencv-python-headless Pillow

# Install project requirements if they exist
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt 2>/dev/null || true
fi

# Install specific packages that might be missing
pip install -q PyMuPDF paddleocr 2>/dev/null || true

echo "Environment ready!"
SETUP_EOF
    
    log_success "DGX environment ready!"
}

# ═══════════════════════════════════════════════════════════════════════════
# Run Streamlit on DGX
# ═══════════════════════════════════════════════════════════════════════════

run_streamlit_dgx() {
    log_info "Starting Streamlit on DGX2..."
    
    # Kill any existing Streamlit processes
    ssh_cmd "pkill -f 'streamlit run' 2>/dev/null || true"
    sleep 2
    
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${GREEN}🚀 Streamlit is starting on DGX2!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${CYAN}   📍 Access URL: ${BOLD}http://${DGX_HOST}:${STREAMLIT_PORT}${NC}"
    echo -e "${CYAN}   📍 Or via Tailscale: ${BOLD}http://spark-5cda:${STREAMLIT_PORT}${NC}"
    echo ""
    echo -e "${YELLOW}   Press Ctrl+C to stop the server${NC}"
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Run Streamlit in foreground (interactive)
    ssh_interactive "cd ~/doc2data && source venv/bin/activate && streamlit run app/streamlit_main.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0 --server.headless true"
}

# ═══════════════════════════════════════════════════════════════════════════
# Run Streamlit in Background on DGX
# ═══════════════════════════════════════════════════════════════════════════

run_streamlit_dgx_bg() {
    log_info "Starting Streamlit on DGX2 (background)..."
    
    # Kill any existing Streamlit processes
    ssh_cmd "pkill -f 'streamlit run' 2>/dev/null || true"
    sleep 2
    
    # Start in background with nohup
    ssh_cmd "cd ~/doc2data && source venv/bin/activate && nohup streamlit run app/streamlit_main.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0 --server.headless true > streamlit.log 2>&1 &"
    
    sleep 3
    
    # Check if running
    if ssh_cmd "pgrep -f 'streamlit run' > /dev/null"; then
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BOLD}${GREEN}🚀 Streamlit is running on DGX2!${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo -e "${CYAN}   📍 Access URL: ${BOLD}http://${DGX_HOST}:${STREAMLIT_PORT}${NC}"
        echo -e "${CYAN}   📍 Or via Tailscale: ${BOLD}http://spark-5cda:${STREAMLIT_PORT}${NC}"
        echo ""
        echo -e "${YELLOW}   To stop: ./run.sh stop${NC}"
        echo -e "${YELLOW}   To view logs: ./run.sh logs${NC}"
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    else
        log_error "Failed to start Streamlit. Check logs with: ./run.sh logs"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# Stop DGX Services
# ═══════════════════════════════════════════════════════════════════════════

stop_dgx() {
    log_info "Stopping services on DGX2..."
    ssh_cmd "pkill -f 'streamlit run' 2>/dev/null || true"
    ssh_cmd "pkill -f 'uvicorn' 2>/dev/null || true"
    log_success "Services stopped!"
}

# ═══════════════════════════════════════════════════════════════════════════
# View Logs
# ═══════════════════════════════════════════════════════════════════════════

view_logs() {
    log_info "Viewing Streamlit logs on DGX2..."
    ssh_interactive "tail -f ~/doc2data/streamlit.log"
}

# ═══════════════════════════════════════════════════════════════════════════
# Check Status
# ═══════════════════════════════════════════════════════════════════════════

check_status() {
    log_info "Checking DGX2 status..."
    echo ""
    
    # Check SSH connectivity
    if ssh_cmd "echo 'connected'" > /dev/null 2>&1; then
        log_success "SSH connection: OK"
    else
        log_error "SSH connection: FAILED"
        return 1
    fi
    
    # Check if Streamlit is running
    if ssh_cmd "pgrep -f 'streamlit run' > /dev/null"; then
        log_success "Streamlit: RUNNING"
        echo -e "   ${CYAN}URL: http://${DGX_HOST}:${STREAMLIT_PORT}${NC}"
    else
        log_warn "Streamlit: NOT RUNNING"
    fi
    
    # Check GPU
    echo ""
    log_info "GPU Status:"
    ssh_cmd "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null || echo "   Could not query GPU"
}

# ═══════════════════════════════════════════════════════════════════════════
# Local Mode
# ═══════════════════════════════════════════════════════════════════════════

run_local() {
    cd "$LOCAL_DIR"
    
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    echo -e "${GREEN}🚀 Starting Streamlit locally...${NC}"
    echo -e "${BLUE}   URL: http://localhost:${STREAMLIT_PORT}${NC}"
    streamlit run app/streamlit_main.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0
}

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

banner

MODE=${1:-deploy}

case $MODE in
    sync)
        sync_to_dgx
        ;;
    setup)
        sync_to_dgx
        setup_dgx_env
        ;;
    run)
        run_streamlit_dgx
        ;;
    run-bg|background)
        run_streamlit_dgx_bg
        ;;
    deploy)
        sync_to_dgx
        setup_dgx_env
        run_streamlit_dgx
        ;;
    deploy-bg)
        sync_to_dgx
        setup_dgx_env
        run_streamlit_dgx_bg
        ;;
    stop)
        stop_dgx
        ;;
    logs)
        view_logs
        ;;
    status)
        check_status
        ;;
    local)
        run_local
        ;;
    *)
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  sync      - Sync files to DGX2"
        echo "  setup     - Sync + setup environment"
        echo "  run       - Run Streamlit on DGX2 (foreground)"
        echo "  run-bg    - Run Streamlit on DGX2 (background)"
        echo "  deploy    - Full deployment (sync + setup + run)"
        echo "  deploy-bg - Full deployment (background)"
        echo "  stop      - Stop services on DGX2"
        echo "  logs      - View Streamlit logs"
        echo "  status    - Check DGX2 status"
        echo "  local     - Run locally"
        exit 1
        ;;
esac
