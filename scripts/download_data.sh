#!/bin/bash

# Usage: ./download_data.sh <output_folder>
# Example: ./download_data.sh ./data

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <output_folder>"
  exit 1
fi

OUTPUT_DIR="$1"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download to output directory
wget -c https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz -O "$OUTPUT_DIR/data.tar.gz"

# Extract in output directory
tar -xvzf "$OUTPUT_DIR/data.tar.gz" -C "$OUTPUT_DIR"
