#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{raw,processed,patterns}
mkdir -p models/checkpoints
mkdir -p data/predictions

echo "Setup completed! Virtual environment is ready."
echo "To activate the virtual environment, run: source venv/bin/activate" 