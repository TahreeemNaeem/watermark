#!/bin/bash

# Fix for torch.library.custom_op error
# This script resolves version incompatibility issues

set -e

echo "================================================"
echo "Fixing Diffusers Installation"
echo "================================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_green() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_red() {
    echo -e "${RED}✗ $1${NC}"
}

# Ensure virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "wmremoval_env" ]; then
        echo "Activating virtual environment..."
        source wmremoval_env/bin/activate
        print_green "Virtual environment activated"
    else
        print_red "Virtual environment not found. Please run install.sh first."
        exit 1
    fi
fi

echo "Uninstalling conflicting packages..."
pip uninstall -y torch torchvision torchaudio diffusers transformers 2>/dev/null || true
print_green "Old packages removed"

echo ""
echo "Installing compatible PyTorch version..."

# Ask for GPU/CPU choice
echo "Select installation type:"
echo "1) GPU with CUDA 11.8 (recommended)"
echo "2) GPU with CUDA 12.1"
echo "3) CPU only"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
        print_green "PyTorch 2.1.0 with CUDA 11.8 installed"
        ;;
    2)
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
        print_green "PyTorch 2.1.0 with CUDA 12.1 installed"
        ;;
    3)
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        print_green "PyTorch 2.1.0 CPU installed"
        ;;
    *)
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
        print_yellow "Invalid choice, defaulting to CUDA 11.8"
        ;;
esac

echo ""
echo "Installing compatible Diffusers version..."
pip install diffusers==0.25.0
print_green "Diffusers 0.25.0 installed"

echo ""
echo "Installing other dependencies..."
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install safetensors==0.4.1
pip install opencv-python-headless==4.8.1.78
pip install Pillow==10.1.0
pip install numpy==1.24.3
print_green "All dependencies installed"

echo ""
echo "Verifying installation..."
python3 << 'EOF'
import sys

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Running in CPU mode")
    
    from diffusers import StableDiffusionInpaintPipeline
    print("✓ Diffusers: Successfully imported")
    
    from transformers import AutoImageProcessor
    print("✓ Transformers: Successfully imported")
    
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
    
    print("\n✓✓✓ Installation fixed successfully! ✓✓✓")
    sys.exit(0)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    sys.exit(1)
EOF