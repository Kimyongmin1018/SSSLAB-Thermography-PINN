# SSSLAB-Thermography-PINN
Physics-Informed Neural Networks for 1D/2D Transient Heat Conduction with Air-Gap Defects

Physics-Informed Neural Network (PINN) framework implemented in PyTorch for solving 1D and 2D transient heat conduction problems with air-gap (void) defects.

The framework:
(1) Uses surface temperature time-series (e.g., from an IR camera) as data.
(2) Embeds the heat conduction PDE + BCs + IC as soft constraints in the loss.
(3) Treats air-gap depth range as trainable parameters and infers them from data.
(4) Supports k-fold cross-validation and automated hyperparameter search via YAML config.

Features

1D and 2D PINN models

1D: temperature field 
𝑇
(
𝑧
,
𝑡
)
T(z,t)

2D: temperature field 
𝑇
(
𝑥
,
𝑧
,
𝑡
)
T(x,z,t)

Solid vs. air properties blended via a smooth transition (air-gap region).

Air-gap defect inversion

Learns defect depth range 
[
𝑧
0
,
𝑧
1
]
[z
0
	​

,z
1
	​

] as trainable parameters.

Uses PDE residual, boundary conditions, and data loss jointly to recover defect location.

Surface temperature dataset loader

Loads CSV files with:

A few metadata rows at the top (ignored automatically).

A wide data block: time, T_pixel1, T_pixel2, ..., T_pixelN.

Averages multiple pixel columns into a single surface temperature time series 
𝑇
(
𝑡
)
T(t).

Returns:

Input: [z, t] (currently z=0 for the surface).

Target: T(t).

Training pipeline

K-fold training using index-based splits.

Combined loss: PDE + IC + BC + data.

Metrics:

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

Hyperparameter search

YAML-based grid search using keys with suffix _search:

e.g., ff_dim_search, num_blocks_search, activation_fn_search.

Iterates over all combinations and reports fold-averaged metrics.

Logging and checkpoints

Console logs streamed into results/<exp>/logs/.

Best model weights saved as best_expXXX_foldYY.pt under results/<exp>/weights/.

TensorBoard logs (loss, metrics, defect parameters, LR) stored under results/<exp>/tensorboard/.
