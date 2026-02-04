#!/bin/bash
# ============================================================================
# Audio Transcription API - Ubuntu Deployment Script
# ============================================================================

set -e

echo "============================================================================"
echo "🚀 Audio Transcription API - Deployment"
echo "============================================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}❌ Please do not run this script as root${NC}"
    exit 1
fi

# ============================================================================
# 1. System Dependencies
# ============================================================================

echo -e "\n${YELLOW}📦 Installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    git \
    nvidia-utils-535 \
    curl

echo -e "${GREEN}✅ System dependencies installed${NC}"

# ============================================================================
# 2. NVIDIA CUDA Check
# ============================================================================

echo -e "\n${YELLOW}🔍 Checking NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected, will use CPU${NC}"
fi

# ============================================================================
# 3. Python Virtual Environment
# ============================================================================

echo -e "\n${YELLOW}🐍 Setting up Python virtual environment...${NC}"

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create new venv
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo -e "${GREEN}✅ Virtual environment created${NC}"

# ============================================================================
# 4. Install Python Dependencies
# ============================================================================

echo -e "\n${YELLOW}📚 Installing Python dependencies...${NC}"
echo "This may take 5-10 minutes..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"

# ============================================================================
# 5. Environment Configuration
# ============================================================================

echo -e "\n${YELLOW}⚙️  Setting up environment...${NC}"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}📝 Created .env file from template${NC}"
    echo -e "${RED}⚠️  IMPORTANT: Edit .env and add your API keys!${NC}"
    echo "   - HUGGINGFACE_TOKEN (required)"
    echo "   - ANTHROPIC_API_KEY (optional)"
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# Create logs directory
mkdir -p logs
echo -e "${GREEN}✅ Logs directory created${NC}"

# ============================================================================
# 6. Test Installation
# ============================================================================

echo -e "\n${YELLOW}🧪 Testing installation...${NC}"

python3 -c "
import torch
import whisper
import pyannote.audio
import fastapi
print('✅ All packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo -e "${GREEN}✅ Installation test passed${NC}"

# ============================================================================
# 7. Systemd Service (Optional)
# ============================================================================

echo -e "\n${YELLOW}🔧 Creating systemd service...${NC}"

SERVICE_FILE="/etc/systemd/system/audio-api.service"
WORK_DIR=$(pwd)
USER=$(whoami)

sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Audio Transcription API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
Environment="PATH=$WORK_DIR/venv/bin"
ExecStart=$WORK_DIR/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo -e "${GREEN}✅ Systemd service created${NC}"

# ============================================================================
# 8. Completion
# ============================================================================

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE${NC}"
echo "============================================================================"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Start the API:"
echo "   sudo systemctl start audio-api"
echo "   sudo systemctl enable audio-api  # Auto-start on boot"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status audio-api"
echo ""
echo "4. View logs:"
echo "   sudo journalctl -u audio-api -f"
echo ""
echo "5. Test API:"
echo "   curl http://localhost:8000/health"
echo ""
echo "6. Access API docs:"
echo "   http://YOUR_SERVER_IP:8000/docs"
echo ""
echo "============================================================================"