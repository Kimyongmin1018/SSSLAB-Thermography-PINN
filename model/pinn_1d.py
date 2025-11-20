# model/pinn_1d.py
from __future__ import annotations

import torch
import torch.nn as nn


class AirGapPINN1D(nn.Module):
    """
    1D plate (0 <= z <= Lz)의 비정상 열전달 + 중간 슬랩 공기층 결함 역추정 PINN.

    • 입력: x = [z, t]
    • 출력: T(z, t)
    • 결함: z0 ~ z1 구간만 공기층(air), 나머지는 solid

    물리식 (단위계 일관하게 가정):
        ρ c ∂T/∂t = ∂z( k ∂T/∂z )

    앞면 z=0:
        k ∂T/∂z = q_pulse(t) + h (T_ext - T) + εσ (T_ext^4 - T^4)

    뒷면 z=Lz:
        k ∂T/∂z = 0 (단열 가정)
    """

    def __init__(
        self,
        net: nn.Module,
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
        self.Lz = float(Lz)
        self.heaviside_eps = heaviside_eps

        # 고정 물성 (buffer로 등록)
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

        # 역추정 대상 파라미터: 중간 결함 시작/끝 z0, z1
        self.z0 = nn.Parameter(torch.tensor(z0_init, dtype=torch.float32))
        self.z1 = nn.Parameter(torch.tensor(z1_init, dtype=torch.float32))

    # -------------------------
    # Helper
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 2) = [z, t]
        """
        return self.net(x)

    def smooth_heaviside(self, s: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.tanh(s / self.heaviside_eps))

    def material_props(self, z: torch.Tensor):
        """
        z: (N, 1)
        공기층: z0 <= z <= z1
        I_void = H(z - z0) * H(z1 - z)
        """
        I1 = self.smooth_heaviside(z - self.z0)     # 0 (z<z0), 1 (z>z0)
        I2 = self.smooth_heaviside(self.z1 - z)     # 1 (z<z1), 0 (z>z1)
        I_void = I1 * I2                            # [z0, z1]에서 1

        k = self.k_s + I_void * (self.k_a - self.k_s)

        rho_c_s = self.rho_s * self.c_s
        rho_c_a = self.rho_a * self.c_a
        rho_c = rho_c_s + I_void * (rho_c_a - rho_c_s)

        return k, rho_c

    # -------------------------
    # Residuals
    # -------------------------
    def pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        PDE residual: ρ c T_t - ∂z(k T_z) = 0
        x: (N,2) = [z, t], interior collocation
        """
        x = x.requires_grad_(True)
        T = self.forward(x)  # (N,1)

        grad_T = torch.autograd.grad(
            T,
            x,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
        )[0]  # (N,2)
        T_z = grad_T[:, 0:1]
        T_t = grad_T[:, 1:2]

        z = x[:, 0:1]
        k, rho_c = self.material_props(z)

        kT_z = k * T_z  # (N,1)
        grad_kTz = torch.autograd.grad(
            kT_z,
            x,
            grad_outputs=torch.ones_like(kT_z),
            create_graph=True,
        )[0]
        d_kTz_dz = grad_kTz[:, 0:1]

        r = rho_c * T_t - d_kTz_dz
        return r

    def ic_residual(self, x_ic: torch.Tensor) -> torch.Tensor:
        """
        초기 조건 residual: T(z, 0) - T_ext = 0
        x_ic: (N,2), t=0
        """
        T = self.forward(x_ic)
        return T - self.T_ext

    def bc_front_residual(self, x_bc: torch.Tensor, q_pulse_fn) -> torch.Tensor:
        """
        앞면 z=0, flux BC:

            k * ∂T/∂z = q_pulse(t)
                      + h (T_ext - T)
                      + eps_r*sigma (T_ext^4 - T^4)

        x_bc: (N,2) where z=0
        q_pulse_fn: callable(t) -> q(t) [W/m^2]
        """
        x_bc = x_bc.requires_grad_(True)
        T = self.forward(x_bc)

        grad_T = torch.autograd.grad(
            T,
            x_bc,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
        )[0]
        T_z = grad_T[:, 0:1]

        z = x_bc[:, 0:1]  # ideally 0
        t = x_bc[:, 1:2]
        k, _ = self.material_props(z)

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
        x_bc: (N,2) where z=Lz
        """
        x_bc = x_bc.requires_grad_(True)
        T = self.forward(x_bc)

        grad_T = torch.autograd.grad(
            T,
            x_bc,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
        )[0]
        T_z = grad_T[:, 0:1]

        z = x_bc[:, 0:1]
        k, _ = self.material_props(z)
        return k * T_z

    def data_loss(self, x_data: torch.Tensor, T_data: torch.Tensor) -> torch.Tensor:
        """
        표면에서 계측된 T 데이터와의 MSE.
        x_data: (B,2), T_data: (B,1)
        """
        T_pred = self.forward(x_data)
        return torch.mean((T_pred - T_data) ** 2)
