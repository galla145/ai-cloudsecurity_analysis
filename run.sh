#!/bin/bash

# ================================
#  AI-Driven Cloud Threat Detection
#  Automated Pipeline Runner
# ================================

echo "---------------------------------------"
echo "  AI Cloud Security Threat Detection"
echo "  Pipeline Starting..."
echo "---------------------------------------"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/6] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/6] Virtual environment already exists."
fi

# Activate environment
echo "[2/6] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "[3/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check dataset exists
DATAFILE="data/cicids2017_combinenew.csv"
if [ ! -f "$DATAFILE" ]; then
    echo "ERROR: Dataset not found at $DATAFILE"
    echo "Please place cicids2017_combinenew.csv in the data/ directory."
    exit 1
fi

# Run preprocessing
echo "[4/6] Running preprocessing..."
python src/preprocess.py --input data/cicids2017_combinenew.csv --output data/processed.pkl

# Train classification models
echo "[5/6] Training models..."
python src/train_classifier.py --input data/processed.pkl --out_dir results/

# Run clustering + anomaly detection
echo "[6/6] Running clustering/anomaly detection..."
python src/cluster_anomalies.py --input data/processed.pkl --out_dir results/

echo "---------------------------------------"
echo " Pipeline completed successfully!"
echo " Results saved in: results/"
echo "---------------------------------------"
