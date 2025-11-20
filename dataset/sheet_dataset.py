# dataset/sheet_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Union

import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SurfaceTemp1DDataset(Dataset):
    """
    1D 표면 온도 시계열 CSV를 로드하는 Dataset.

    가정:
      - CSV 상단에는 메타데이터/설정 정보가 몇 줄 있을 수 있고(컬럼 수가 적음),
      - 실제 데이터 블록은 "컬럼 수가 많은" 행에서 시작.
      - 실제 데이터 블록은 다음과 같이 구성:
          time, T(x1), T(x2), ..., T(xN)
        -> time = 첫 번째 컬럼
        -> T(t) = 나머지 컬럼들의 평균 (또는 특정 픽셀로 바꾸고 싶으면 여기만 손보면 됨)

    이 Dataset은 PINN 학습을 위해
      x = [z, t],  (여기서 z=0)
      y = T(t)
    형태로 반환합니다.
    """

    def __init__(self, csv_path: Union[str, Path], min_cols_for_data: int = 3) -> None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # 1) "컬럼 수가 많은" 첫 번째 행을 찾아서 실제 데이터 시작 위치(auto-detect)
        data_start_row = self._detect_data_start_row(path, min_cols_for_data)

        # 2) 해당 행을 헤더로 사용해서 pandas로 읽기
        #    -> data_start_row 이전의 줄은 모두 skip
        #    -> data_start_row 행은 header로 쓰이고, 그 다음 행부터가 numeric 데이터
        df = pd.read_csv(path, skiprows=data_start_row)

        if df.shape[1] < 2:
            raise ValueError(
                f"CSV {path} 에서 최소 2개 컬럼(time + temperature...)이 필요하지만 "
                f"{df.shape[1]}개만 발견했습니다."
            )

        # 3) time, T(t) 생성
        # time: 첫 번째 컬럼
        try:
            t = df.iloc[:, 0].to_numpy(dtype=np.float32)
        except Exception as e:
            raise ValueError(
                f"첫 번째 컬럼을 float32로 변환하는 중 오류 발생. "
                f"CSV 구조를 확인해야 합니다. 원인: {e}"
            )

        # 온도: 나머지 컬럼들 (여러 픽셀) 평균 -> 하나의 T(t)
        if df.shape[1] == 2:
            # time, T 한 컬럼만 있을 때
            T_vals = df.iloc[:, 1].to_numpy(dtype=np.float32)
        else:
            # time, T(x1..xN) 여러 컬럼 -> 평균
            T_vals = df.iloc[:, 1:].to_numpy(dtype=np.float32).mean(axis=1)

        # 4) PINN 입력 x = [z, t] (여기서 z=0만 사용)
        z = np.zeros_like(t, dtype=np.float32)
        x = np.stack([z, t], axis=1)  # (N, 2)

        self.x = torch.from_numpy(x)             # (N,2)
        self.T = torch.from_numpy(T_vals).unsqueeze(1)  # (N,1)
        self.path = str(path)

        print(
            f"[SurfaceTemp1DDataset] Loaded {path} | "
            f"data_start_row={data_start_row} | "
            f"N={len(self)}, num_cols={df.shape[1]}"
        )

    @staticmethod
    def _detect_data_start_row(path: Path, min_cols_for_data: int) -> int:
        """
        CSV 파일 상단의 메타데이터 줄(컬럼 수가 적은 줄)을 건너뛰고,
        '컬럼 수가 min_cols_for_data 이상'인 첫 행을 데이터 헤더로 간주.

        예) 에러 메시지:
            Expected 2 fields in line 9, saw 504
            -> 앞의 몇 줄(1~8)은 2개 필드, 9번째 줄은 504개 필드
            -> 504개 필드가 있는 9번째 줄부터가 실제 데이터 블록으로 보는 것이 합리적
        """
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                # 빈 줄은 스킵
                if not row:
                    continue
                # 컬럼 수가 min_cols_for_data 이상이면 데이터 시작으로 간주
                if len(row) >= min_cols_for_data:
                    return i
        # 혹시 못 찾으면 파일 전체를 데이터로 가정
        return 0

    def __len__(self) -> int:  # type: ignore[override]
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        # x: (2,), T: (1,)
        return self.x[idx], self.T[idx]


# ----------------------------
# 기존 코드와 호환용 alias
# ----------------------------

SurfaceTemperatureDataset = SurfaceTemp1DDataset


def load_surface_npz(csv_path: Union[str, Path]) -> SurfaceTemp1DDataset:
    """
    예전 함수 이름(load_surface_npz)과 호환시키기 위한 래퍼.
    실제로는 npz가 아니라 CSV를 로드합니다.
    """
    return SurfaceTemp1DDataset(csv_path)
