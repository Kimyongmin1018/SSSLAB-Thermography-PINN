# model/pinn_2d.py
from __future__ import annotations

import torch
import torch.nn as nn


class AirGapPINN2D(nn.Module):
    """
    2D PINN: T(x, z, t)

    좌표:
        x[:, 0] = x (수평 좌표)
        x[:, 1] = z (깊이)
        x[:, 2] = t (시간)

    결함(공기층)은 z0 ~ z1 구간에서만 존재하는 slab 형태로 가정
    (x 방향으로는 균일).

    PDE (에너지 보존식):
        rho c * T_t = d/dx(k T_x) + d/dz(k T_z)

    경계 조건:
        - z = 0 (앞면):  k T_z = q_pulse(t) + h (T_ext - T) + eps*sigma (T_ext^4 - T^4)
        - z = Lz (뒷면): k T_z = 0  (단열)
    """

    def __init__(
        self,
        net: nn.Module,
        Lx: float,
        Lz: float,
        k_s: float,
        rho_s: float,
        c_s: float,
        k_a: float,
        rho_a: float,
        c_a: float,
        T_ext: float,
        h_front: float,
        eps_r: float,
        sigma: float,
        z0_init: float,
        z1_init: float,
        heaviside_eps: float = 1e-3,
    ) -> None:
        super().__init__()

        self.net = net
        self.Lx = float(Lx)
        self.Lz = float(Lz)
        self.heaviside_eps = float(heaviside_eps)

        # 고체/공기 물성 (buffer 로 등록)
        self.register_buffer("k_s", torch.tensor(k_s, dtype=torch.float32))
        self.register_buffer("rho_s", torch.tensor(rho_s, dtype=torch.float32))
        self.register_buffer("c_s", torch.tensor(c_s, dtype=torch.float32))

        self.register_buffer("k_a", torch.tensor(k_a, dtype=torch.float32))
        self.register_buffer("rho_a", torch.tensor(rho_a, dtype=torch.float32))
        self.register_buffer("c_a", torch.tensor(c_a, dtype=torch.float32))

        self.register_buffer("T_ext", torch.tensor(T_ext, dtype=torch.float32))
        self.register_buffer("h_front", torch.tensor(h_front, dtype=torch.float32))
        self.register_buffer("eps_r", torch.tensor(eps_r, dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

        # 역추정 대상: 결함 깊이 범위 [z0, z1]
        self.z0 = nn.Parameter(torch.tensor(z0_init, dtype=torch.float32))
        self.z1 = nn.Parameter(torch.tensor(z1_init, dtype=torch.float32))

    # -------------------------
    # Helper
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 3) = [x, z, t]
        """
        return self.net(x)

    def smooth_heaviside(self, s: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.tanh(s / self.heaviside_eps))

    def material_props(self, x: torch.Tensor):
        """
        x: (N, 3), 여기서 두 번째 컬럼 x[:, 1] = z
        공기층: z0 <= z <= z1
        """
        z = x[:, 1:2]

        I1 = self.smooth_heaviside(z - self.z0)     # 0 (z<z0), 1 (z>z0)
        I2 = self.smooth_heaviside(self.z1 - z)     # 1 (z<z1), 0 (z>z1)
        I_void = I1 * I2                            # [z0, z1]에서 1

        # 열전도율
        k = self.k_s + I_void * (self.k_a - self.k_s)

        # 밀도*비열
        rho_c_s = self.rho_s * self.c_s
        rho_c_a = self.rho_a * self.c_a
        rho_c = rho_c_s + I_void * (rho_c_a - rho_c_s)

        return k, rho_c

    # -------------------------
    # Residuals
    # -------------------------
    def pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        PDE residual:
            rho c T_t - [ d/dx(k T_x) + d/dz(k T_z) ] = 0

        x: (N,3) = [x, z, t], domain 내부 collocation points
        """
        x = x.requires_grad_(True)
        T = self.forward(x)  # (N,1)

        grad_T = torch.autograd.grad(
            T,
            x,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
        )[0]  # (N,3)

        T_x = grad_T[:, 0:1]
        T_z = grad_T[:, 1:2]
        T_t = grad_T[:, 2:3]

        k, rho_c = self.material_props(x)

        kT_x = k * T_x
        kT_z = k * T_z

        grad_kT_x = torch.autograd.grad(
            kT_x,
            x,
            grad_outputs=torch.ones_like(kT_x),
            create_graph=True,
        )[0]
        grad_kT_z = torch.autograd.grad(
            kT_z,
            x,
            grad_outputs=torch.ones_like(kT_z),
            create_graph=True,
        )[0]

        d_kT_x_dx = grad_kT_x[:, 0:1]
        d_kT_z_dz = grad_kT_z[:, 1:2]

        r = rho_c * T_t - (d_kT_x_dx + d_kT_z_dz)
        return r

    def ic_residual(self, x_ic: torch.Tensor) -> torch.Tensor:
        """
        초기 조건 residual: T(x, z, t=0) - T_ext = 0
        x_ic: (N,3), t=0
        """
        T = self.forward(x_ic)
        return T - self.T_ext

    def bc_front_residual(self, x_bc: torch.Tensor, q_pulse_fn) -> torch.Tensor:
        """
        앞면 z=0 flux BC:
            k * ∂T/∂z = q_pulse(t)
                      + h (T_ext - T)
                      + eps_r*sigma (T_ext^4 - T^4)

        x_bc: (N,3), 두 번째 컬럼 z=0 으로 샘플링
        """
        x_bc = x_bc.requires_grad_(True)
        T = self.forward(x_bc)

        grad_T = torch.autograd.grad(
            T,
            x_bc,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
        )[0]
        T_z = grad_T[:, 1:2]  # z 방향 미분

        k, _ = self.material_props(x_bc)
        t = x_bc[:, 2:3]

        lhs = k * T_z
        q_pulse = q_pulse_fn(t)

        rhs = (
            q_pulse
            + self.h_front * (self.T_ext - T)
            + self.eps_r * self.sigma * (self.T_ext**4 - T**4)
        )

        return lhs - rhs

    def bc_back_residual(self, x_bc: torch.Tensor) -> torch.Tensor:
        """
        뒷면 z=Lz, adiabatic: k ∂T/∂z = 0
        x_bc: (N,3), 두 번째 컬럼 z=Lz 로 샘플링
        """
        x_bc = x_bc.requires_grad_(True)
        T = self.forward(x_bc)

        grad_T = torch.autograd.grad(
            T,
            x_bc,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
        )[0]
        T_z = grad_T[:, 1:2]

        k, _ = self.material_props(x_bc)
        return k * T_z

    def data_loss(self, x_data: torch.Tensor, T_data: torch.Tensor) -> torch.Tensor:
        """
        표면 (혹은 임의 위치) 데이터와의 MSE
        x_data: (B,3), T_data: (B,1)
        """
        T_pred = self.forward(x_data)
        return torch.mean((T_pred - T_data) ** 2)
