#!/bin/bash



# Default choices if you want compare Fig6B
epochs1=200
epochs2=5000
epochs3=500

# Edit to override the defaults

#epochs1=
#epochs2=
#epochs3=

python analyze.py --dataset mnist --epochs $epochs1

python analyze.py --dataset mnist_grad --epochs $epochs2

python analyze.py --dataset mnist_shuffled --epochs $epochs3