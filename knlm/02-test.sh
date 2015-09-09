#!/bin/bash -e

#TRAINSET=../data/morph/finnish.clean.train10k
TRAINSET=../data/morph/finnish.clean.train
VALIDSET=../data/morph/finnish.clean.test
MODEL=finnish-kn.4bo

ngram -order 4 -lm "$MODEL" -ppl "$VALIDSET" -debug 0 >"$MODEL.ppl"
