#!/bin/bash

# Standard dataset
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O train.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O dev.json

# Unzip compressed dataset
zip -s 0 augmented_data/augmented_zips.zip --out augmented_train.json.zip
unzip augmented_train.json.zip
