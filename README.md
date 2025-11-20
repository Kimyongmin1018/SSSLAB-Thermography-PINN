# SSSLAB-Thermography-PINN
Physics-Informed Neural Networks for 1D/2D Transient Heat Conduction with Air-Gap Defects

Physics-Informed Neural Network (PINN) framework implemented in PyTorch for solving 1D and 2D transient heat conduction problems with air-gap (void) defects.

The framework:
(1) Uses surface temperature time-series (e.g., from an IR camera) as data.
(2) Embeds the heat conduction PDE + BCs + IC as soft constraints in the loss.
(3) Treats air-gap depth range as trainable parameters and infers them from data.
(4) Supports k-fold cross-validation and automated hyperparameter search via YAML config.
