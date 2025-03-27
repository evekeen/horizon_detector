#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 input-dir"
    exit 1
fi

input_dir="$1"

if [ ! -d "$input_dir" ]; then
    echo "Error: Directory $input_dir does not exist"
    exit 1
fi

if [ -f "$input_dir/metadata.csv" ]; then
    cp "$input_dir/metadata.csv" ./
else
    echo "Warning: metadata.csv not found in input directory"
fi

for f in "$input_dir"/*.zip; do
    [ -f "$f" ] || continue
    unzip "$f"
done