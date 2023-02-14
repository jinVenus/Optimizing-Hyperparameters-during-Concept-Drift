# Optimizing-Hyperparameters-during-Concept-Drift
Optimizing-Hyperparameters-during-Concept-Drift:

This project is based on Model-based Optimization on Distributed Embedded System (https://github.com/Strange369/MODES-public). And it is mainly used to explore whether the model needs to re-choose new hyperparameters based on the new data when concept drift occurs in the data.

The implementation of MODES and corresponding experiments are released here:

The botorch-modes implementation is based on BoTorch (https://botorch.org/).
The model regression function is based on the tutorial (https://botorch.org/tutorials/closed_loop_botorch_only).

The current experiments are mainly done with MNIST dataset and adding different levels of fault rate into it.
