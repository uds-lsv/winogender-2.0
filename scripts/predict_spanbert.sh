#!/usr/bin/env bash

export data_dir="/nethome/jsteuer/git/winogender-2.0/data/spanbert"
export STIMULI_DIR="/nethome/jsteuer/git/winogender-2.0/spanbert"
export SPANBERT_DIR="/nethome/jsteuer/git/coref"
export PYTHON_BIN="/nethome/jsteuer/miniconda3/envs/spanbert/bin"

cd $SPANBERT_DIR

GPU=0 $PYTHON_BIN/python $SPANBERT_DIR/predict.py spanbert_base $STIMULI_DIR/single.jsonlines $STIMULI_DIR/single_base_res.jsonlines
GPU=0 $PYTHON_BIN/python $SPANBERT_DIR/predict.py spanbert_base $STIMULI_DIR/double.jsonlines $STIMULI_DIR/double_base_res.jsonlines
GPU=0 $PYTHON_BIN/python $SPANBERT_DIR/predict.py spanbert_base $STIMULI_DIR/double_old.jsonlines $STIMULI_DIR/double_old_base_res.jsonlines
GPU=0 $PYTHON_BIN/python $SPANBERT_DIR/predict.py spanbert_large $STIMULI_DIR/single.jsonlines $STIMULI_DIR/single_large_res.jsonlines
GPU=0 $PYTHON_BIN/python $SPANBERT_DIR/predict.py spanbert_large $STIMULI_DIR/double.jsonlines $STIMULI_DIR/double_large_res.jsonlines
GPU=0 $PYTHON_BIN/python $SPANBERT_DIR/predict.py spanbert_large $STIMULI_DIR/double_old.jsonlines $STIMULI_DIR/double_old_large_res.jsonlines
