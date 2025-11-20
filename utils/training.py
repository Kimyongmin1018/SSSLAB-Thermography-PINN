# utils/training.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import optim

from model.pinn_1d import AirGapPINN1D


def default_q_pulse(t: torch.Tensor, q0: float = 5e4, t_start: float = 0.0, t_end: float = 0.1) -> torch.Tensor:
    """Square heat flux pulse in time."""
    # t is (N,1)
    cond = (t >= t_start) & (t <= t_end)
    return q0 * cond.to(t.dtype)


def train_pinn_1d(
    model: AirGapPINN1D,
    t_data: np.ndarray,
    T_data: np.ndarray,
    cfg: Dict,
    device: torch.device,
) -> Tuple[AirGapPINN1D, Dict[str, float]]:
    """Train 1D air-gap PINN on a single surface temperature time-series.

    Parameters
    ----------
    model:
        AirGapPINN1D instance.
    t_data, T_data:
        1D numpy arrays with same shape, surface measurement at z=0.
    cfg:
        Parsed YAML config dict.
    device:
        torch.device to use.

    Returns
    -------
    model:
        Trained model (on `device`).
    history:
        Dictionary with final losses and learned defect depth.
    """
    model.to(device)

    # Training hyperparameters
    total_steps = int(cfg.get("total_steps", 3000))
    eval_interval = int(cfg.get("eval_interval", 100))
    batch_size = int(cfg.get("batch_size", 128))
    hparams = cfg.get("hparams", {})
    lr = float(hparams.get("lr", 1e-3))

    # Physics loss weights (can be moved to config if desired)
    w_pde = float(hparams.get("w_pde", 1.0))
    w_ic = float(hparams.get("w_ic", 1.0))
    w_bc_front = float(hparams.get("w_bc_front", 1.0))
    w_bc_back = float(hparams.get("w_bc_back", 1.0))
    w_data = float(hparams.get("w_data", 1.0))

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert measurement data to tensors
    t_data_t = torch.from_numpy(t_data.astype(np.float32)).to(device).view(-1, 1)
    T_data_t = torch.from_numpy(T_data.astype(np.float32)).to(device).view(-1, 1)

    Tmax = float(t_data_t.max().item())
    Lz = float(model.Lz)

    # Collocation sizes
    num_pde = int(hparams.get("num_pde_points", 1024))
    num_ic = int(hparams.get("num_ic_points", 256))
    num_bc = int(hparams.get("num_bc_points", 256))

    history: Dict[str, float] = {}

    for step in range(1, total_steps + 1):
        model.train()
        optimizer.zero_grad()

        # ---------------------------
        # PDE residual points (interior)
        # ---------------------------
        z_f = torch.rand(num_pde, 1, device=device) * Lz
        t_f = torch.rand(num_pde, 1, device=device) * Tmax
        r_pde = model.pde_residual(z_f, t_f)
        loss_pde = torch.mean(r_pde**2)

        # ---------------------------
        # Initial condition residual T(z,0) = T_ext
        # ---------------------------
        z_ic = torch.rand(num_ic, 1, device=device) * Lz
        r_ic = model.ic_residual(z_ic)
        loss_ic = torch.mean(r_ic**2)

        # ---------------------------
        # Boundary conditions
        # ---------------------------
        t_bc = torch.rand(num_bc, 1, device=device) * Tmax

        r_bc_front = model.bc_front_residual(t_bc, default_q_pulse)
        loss_bc_front = torch.mean(r_bc_front**2)

        r_bc_back = model.bc_back_residual(t_bc)
        loss_bc_back = torch.mean(r_bc_back**2)

        # ---------------------------
        # Data loss at surface z=0
        # ---------------------------
        if batch_size > t_data_t.shape[0]:
            batch_size_eff = t_data_t.shape[0]
        else:
            batch_size_eff = batch_size
        idx = torch.randint(0, t_data_t.shape[0], (batch_size_eff,), device=device)
        t_batch = t_data_t[idx]
        T_batch = T_data_t[idx]
        z_batch = torch.zeros_like(t_batch)

        T_pred = model(z_batch, t_batch)
        loss_data = torch.mean((T_pred - T_batch) ** 2)

        # ---------------------------
        # Total loss and optimization
        # ---------------------------
        loss = (
            w_pde * loss_pde
            + w_ic * loss_ic
            + w_bc_front * loss_bc_front
            + w_bc_back * loss_bc_back
            + w_data * loss_data
        )

        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == 1 or step == total_steps:
            print(
                f"[step {step:6d}] "
                f"loss={loss.item():.4e} | "
                f"pde={loss_pde.item():.4e}, ic={loss_ic.item():.4e}, "
                f"bc_f={loss_bc_front.item():.4e}, bc_b={loss_bc_back.item():.4e}, "
                f"data={loss_data.item():.4e}, z_d={model.z_d.item():.4e}"
            )

    history["loss"] = float(loss.item())
    history["loss_pde"] = float(loss_pde.item())
    history["loss_ic"] = float(loss_ic.item())
    history["loss_bc_front"] = float(loss_bc_front.item())
    history["loss_bc_back"] = float(loss_bc_back.item())
    history["loss_data"] = float(loss_data.item())
    history["z_d"] = float(model.z_d.item())

    # Save final model
    save_root = Path(cfg.get("result_save_root", "results"))
    save_root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_root / "pinn_1d_final.pt")

    return model, history
