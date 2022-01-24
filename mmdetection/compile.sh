#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

echo "Building roi align op..."
cd mmdet/ops/roi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi pool op..."
cd ../roi_pool
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building nms op..."
cd ../nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

# echo "Building kpts nms op..."
# cd ../kpts_nms
# if [ -d "build" ]; then
#     rm -r build
# fi
# $PYTHON setup.py build_ext --inplace

echo "Building dcn..."
cd ../dcn
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building sigmoid focal loss op..."
cd ../sigmoid_focal_loss
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building masked conv op..."
cd ../masked_conv
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
