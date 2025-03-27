#!/bin/bash

purge=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--purge)
            purge=true
            shift
            ;;
        *)
            input_dir="$1"
            shift
            ;;
    esac
done

if [ -z "$input_dir" ]; then
    echo "Usage: $0 [-p|--purge] input-dir"
    echo "Options:"
    echo "  -p, --purge    Delete zip files after unpacking"
    exit 1
fi

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
    if [ "$purge" = true ]; then
        rm "$f"
    fi
done