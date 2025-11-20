# model/__init__.py

from .pinn_1d import AirGapPINN1D
from .pinn_2d import AirGapPINN2D   # 2D 모델은 pinn_2d.py에서 가져와야 함
from .layers import MLP             # 필요하면 같이 노출


__all__ = ["AirGapPINN1D", "AirGapPINN2D"]
