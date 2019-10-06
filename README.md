LICENSE

## Intrinsic dimension of data representations in deep neural networks


This repository contains the code to reproduce the results of the following paper

["Intrinsic dimension of data representations in deep neural networks"](https://arxiv.org/abs/1905.12784). _Neural Information Processing Systems (NeurIPS) 2019_

Morover we provide [jupyter notebook tutorials](https://github.com/ansuini/IntrinsicDimDeep/tree/master/tutorials) where we describe more in detail the methodology used and point to possible extensions and open problems.

We were largely inspired from the amazing repository ["(SV)CCA"](https://github.com/google/svcca) for the structuring of this README.

### Code Structure and Usage

[rephrase]
The [tutorials](https://github.com/ansuini/IntrinsicDimDeep/tree/master/tutorials) overview all of the main scripts, provide an example implementation, and also discuss existing applications and new directions. 


The main script is [cca_core](https://github.com/google/svcca/blob/master/cca_core.py) which can be used to compute CCA between two neural networks and outputs both the CCA correlation coefficients as well as the CCA directions. The CCA for Conv Layers tutorial outlines applications to convolutional layers.
