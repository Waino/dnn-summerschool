#!/bin/bash -e

#TRAINSET=../data/morph/finnish.clean.train10k
TRAINSET=../data/morph/finnish.clean.train
VALIDSET=../data/morph/finnish.clean.test
MODEL=finnish-kn.4bo

ngram-count \
          -order 4 \
          -interpolate1 -interpolate2 -interpolate3 -interpolate4 \
          -kndiscount1 -kndiscount2 -kndiscount3 -kndiscount4 \
          -gt4min 2 \
          -limit-vocab \
          -text "$TRAINSET" \
          -lm "$MODEL" \
          $*
