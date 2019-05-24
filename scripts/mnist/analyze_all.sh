#!/bin/bash



# Default choices if you want compare Fig6B
epochs1=200
epochs2=5000
epochs3=500

# Edit to override the defaults

#epochs1=
#epochs2=
#epochs3=

python analyze.py --dataset mnist --epoch $epoch1

python analyze.py --dataset mnist_grad --epoch $epoch2

python analyze.py --dataset mnist_shuffled --epoch $epoch3