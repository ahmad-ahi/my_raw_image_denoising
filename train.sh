#!/bin/bash

echo "=== Job started on $(hostname) at $(date) ==="

if [ ! -d "raw_image_denoising" ]; then
    echo "Extracting code..."
    tar xzf code_v5.tar.gz
    rm -f code_v5.tar.gz

fi

pip install --no-cache-dir --user -r raw_image_denoising/requirements.txt

if [ ! -d "data/SID" ] || [ ! -d "data/ELD" ]; then
    echo "Extracting dataset (first run)..."
    mkdir -p data
    tar xzf sid_eld.tar.gz -C data
    rm -f sid_eld.tar.gz
else
    echo "Dataset already extracted, skipping."
fi


mkdir -p raw_image_denoising/checkpoints
if [ -d "improved" ]; then
    echo "Found existing checkpoint directory, placing it for resume..."
    cp -r improved raw_image_denoising/checkpoints/improved
fi

cd raw_image_denoising

mkdir -p checkpoints/improved

timeout 4h python3 train.py \
    --model nafnet \
    --info_path ./infos/ELD_SonyA7S2.info \
    --dark_shading_dir ./resources/SonyA7S2 \
    --save_dir checkpoints/improved \
    --epochs 1000 \
    --save_every 200 \
    --device cuda:0

status=$?
echo "=== train.py exited with status $status at $(date) ==="

if [ $status -eq 124 ]; then
    echo "Hit the 4-hour timeout — requesting HTCondor requeue (exit 85)."
    exit 85
fi

exit $status
