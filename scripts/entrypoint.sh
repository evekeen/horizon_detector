#!/bin/bash
set -e

# Run authentication setup
source /workspace/setup_auth.sh

# Clone the repository
echo "Cloning horizon detector repository..."
git clone https://github.com/evekeen/horizon_detector.git /workspace/repo
cd /workspace/repo

# Download dataset from S3 if S3_DATASET_PATH is provided
if [ ! -z "$S3_DATASET_PATH" ]; then
    echo "Downloading dataset from $S3_DATASET_PATH..."
    mkdir -p /workspace/data
    aws s3 sync $S3_DATASET_PATH /workspace/data
    
    echo "Unpacking dataset..."
    cd /workspace/data
    /workspace/unpack_dataset.sh .
    cd /workspace/repo
fi

# Keep container running
echo "Setup completed. Container kept running for SSH access."
tail -f /dev/null