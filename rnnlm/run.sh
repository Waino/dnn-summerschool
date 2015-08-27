#!/bin/bash -e

source ../../bin/activate
export THEANO_FLAGS=floatX=float32,device=gpu
python train_lm.py
