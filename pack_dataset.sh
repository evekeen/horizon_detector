#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 output-dir"
    exit 1
fi

output_dir="$1"
mkdir -p "$output_dir"

if [ -f "metadata.csv" ]; then
    cp metadata.csv "$output_dir/"
else
    echo "Warning: metadata.csv not found in current directory"
fi

for d in images/*/; do
  [ -d "$d" ] || continue
  bn=$(basename "$d")
  zip -r -0 "$output_dir/${bn}.zip" "$d"
done