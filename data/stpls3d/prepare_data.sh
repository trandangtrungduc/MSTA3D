#!/bin/bash
echo Preprocess data
python prepare_data_inst.py
echo Prepare superpoints
python prepare_superpoint.py