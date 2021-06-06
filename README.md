# Introduction
This repository holds the code for the final project of CS 583 - Introduction to Computer Vision.
The goal of this repository is to train a control and a Bayesian "experimental" version of two different architectures on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
These architectures are then evaluated for adversarial robustness using a set of techniques from the [Counterfit](https://github.com/Azure/counterfit) project.

# Instructions
In order to run the experiment, you will need to install `tensorflow` v2.1 or above and follow the instructions to install [Counterfit](https://github.com/Azure/counterfit)
To recreate the results:
1. Run `python model.py`
2. Use the instructions from the Counterfit wiki and the targets files in the `/src/attacks/` repo to add the targets to Counterfit
3. run `scan -l -v -c -n 100` on each of the target models using Counterfit.
