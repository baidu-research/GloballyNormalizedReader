# This file must be sourced, or copied into a ~/.bashrc or ~/.zshrc file.

# Use Python 3.6 (adds the python3.6 command).
export PATH=/tools/python3.6/bin:$PATH

# CUDA version.
export CUDA_PATH=/tools/cuda-8.0.61
export CUDA_HOME=${CUDA_PATH}

# CuDNN version.
export CUDNN_PATH=/tools/cudnn-8.0-linux-x64-v6.0

# Avoid long output messages when TF OOM's.
export TF_CPP_MIN_LOG_LEVEL='3'

