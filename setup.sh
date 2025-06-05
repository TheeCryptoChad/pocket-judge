#!/bin/bash

# Detect platform
UNAME=$(uname | tr '[:upper:]' '[:lower:]')

# Decide on python executable
if [[ "$UNAME" == *"mingw"* || "$UNAME" == *"msys"* || "$UNAME" == *"windows"* ]]; then
    PYTHON="python"
    ACTIVATE_PATH="venv/Scripts/activate"
else
    PYTHON="python3"
    ACTIVATE_PATH="venv/bin/activate"
fi

# Step 1: Create virtual environment
echo "🔧 Creating virtual environment..."
$PYTHON -m venv venv

# Step 2: Activate virtual environment
echo "⚡ Activating virtual environment..."
source $ACTIVATE_PATH

# Step 3: Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Step 4: Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 5: Load dataset
echo "📁 Running dataset loader..."
$PYTHON ./models/pose/scripts/load_dataset.py

# Done!
echo "✅ Setup complete!"
echo "👉 To activate the virtual environment manually, run: source $ACTIVATE_PATH"