#!/bin/bash


# quick setup for imaginecraft model build venv

# Slurm/watcloud suited venv
build_venv() {
    export HF_HOME=/tmp/huggingface # Huggingface model cache to prevent quota issues

    echo "Starting virtual environment build process..."
    
    # Dummy command 1: Check Python version
    echo "Checking Python version..."
    python3 --version
    sleep 1
    
    # Dummy command 2: Create virtual environment
    echo "Creating virtual environment..."
    python3 -m venv /tmp/venv
    sleep 1

    source /tmp/venv/bin/activate
    sleep 1
    
    pip3 install --upgrade pip
    sleep 1

    # Special cuda torch wheels
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    sleep 1

    pip3 install -r requirements.txt
    sleep 1

    # Prompt hf login
    huggingface-cli login

    echo "Virtual environment build completed successfully!"
}