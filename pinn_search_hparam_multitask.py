# pinn_search_hparam_multitask.py
import sys
import os
import argparse
import datetime
import shutil
import time
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset.sheet_dataset import SurfaceTemp1DDataset
from model.layers import MLP
from model.pinn_1d import AirGapPINN1D
from model.pinn_2d import AirGapPINN2D

from utils.config import load_config
from utils.seed import set_random_seed


# =========================
# Logger: stdout + 파일 로그
# =========================
class Logger(object):
    def __init__(self, filename: str = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1, encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_directories(cfg, exp_name: str):
    result_save_root = cfg["result_save_root"]

    log_dir = os.path.join(result_save_root, exp_name, "logs")
    weight_dir = os.path.join(result_save_root, exp_name, "weights")
    tb_dir = os.path.join(result_save_root, exp_name, "tensorboard")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    )
    sys.stdout = Logger(log_path)

    return log_dir, weight_dir, tb_dir


# =========================
# Hyper-parameter 조합
# =========================
def get_hyperparameter_combinations(hparams: dict):
    search_keys = [k for k in hparams if k.endswith("_search")]

    # search key 없으면 그대로 한 번만
    if not search_keys:
        return [hparams]

    base = {k: v for k, v in hparams.items() if not k.endswith("_search")}
    search_values = [hparams[k] for k in search_keys]

    combinations = []
    for combo in product(*search_values):
        params = dict(base)
        for key, value in zip(search_keys, combo):
            base_key = key[:-7]  # "_search" 제거
            params[base_key] = value
        combinations.append(params)
    return combinations


# =========================
# 단순 K-fold 분할 (index 기준)
# =========================
def make_kfold_indices(n_samples: int, kfold_n: int):
    indices = np.arange(n_samples)

    # kfold_n <= 1: 모든 데이터를 train/valid 둘 다 사용
    if kfold_n <= 1:
        return [(indices, indices)]

    fold_sizes = [n_samples // kfold_n] * kfold_n
    for i in range(n_samples % kfold_n):
        fold_sizes[i] += 1

    folds = []
    current = 0
    for fs in fold_sizes:
        start, stop = current, current + fs
        valid_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, valid_idx))
        current = stop
    return folds


# =========================
# q_pulse: square pulse 예시
# =========================
def q_pulse_square(t: torch.Tensor, q0: float, t_start: float, t_end: float) -> torch.Tensor:
    """
    t: (N,1)
    q0: [W/m^2]
    """
    mask = (t >= t_start) & (t <= t_end)
    return q0 * mask.to(t.dtype)


# =========================
# 모델/optimizer 생성
# =========================
def build_model_from_params(params: dict, cfg: dict, device: torch.device):
    # 모델 차원(1D/2D)
    # 우선순위: cfg["model_dimension"] > params["model_dimension"] > 기본값 1
    model_dim = int(cfg.get("model_dimension", params.get("model_dimension", 1)))

    # 네트워크 구조
    ff_dim = int(params.get("ff_dim", 128))
    num_blocks = int(params.get("num_blocks", 4))
    activation_fn = params.get("activation_fn", "tanh")

    in_dim = 2 if model_dim == 1 else 3  # 1D: (z,t), 2D: (x,z,t)

    net = MLP(
        in_dim=in_dim,
        out_dim=1,
        hidden_dim=ff_dim,
        num_hidden_layers=num_blocks,
        activation=activation_fn,
    )

    # 공통 물리 파라미터
    Lz = float(cfg.get("Lz", 0.01))
    Lx = float(cfg.get("Lx", 0.01))  # 2D 에서 사용

    k_s = float(cfg.get("k_s", 20.0))
    rho_s = float(cfg.get("rho_s", 7800.0))
    c_s = float(cfg.get("c_s", 500.0))

    k_a = float(cfg.get("k_a", 0.026))
    rho_a = float(cfg.get("rho_a", 1.2))
    c_a = float(cfg.get("c_a", 1000.0))

    T_ext = float(cfg.get("T_ext", 293.15))
    h_front = float(cfg.get("h_front", 20.0))
    eps_r = float(cfg.get("eps_r", 1.0))
    sigma = float(cfg.get("sigma", 5.670374419e-8))

    z0_init = float(cfg.get("z0_init", 0.3 * Lz))
    z1_init = float(cfg.get("z1_init", 0.7 * Lz))

    # 모델 생성
    if model_dim == 1:
        model = AirGapPINN1D(
            net=net,
            Lz=Lz,
            k_s=k_s,
            rho_s=rho_s,
            c_s=c_s,
            k_a=k_a,
            rho_a=rho_a,
            c_a=c_a,
            T_ext=T_ext,
            h_front=h_front,
            eps_r=eps_r,
            sigma=sigma,
            z0_init=z0_init,
            z1_init=z1_init,
        ).to(device)
    elif model_dim == 2:
        model = AirGapPINN2D(
            net=net,
            Lx=Lx,
            Lz=Lz,
            k_s=k_s,
            rho_s=rho_s,
            c_s=c_s,
            k_a=k_a,
            rho_a=rho_a,
            c_a=c_a,
            T_ext=T_ext,
            h_front=h_front,
            eps_r=eps_r,
            sigma=sigma,
            z0_init=z0_init,
            z1_init=z1_init,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_dimension: {model_dim}")

    # Optimizer
    lr = float(params.get("lr", 1e-3))
    optim_name = params.get("optimizer_name", "AdamW")

    if optim_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, optimizer, model_dim


# =========================
# 단일 실험 (여러 k-fold)
# =========================
def run_experiment(
    params: dict,
    all_search_fields,
    dataset: SurfaceTemp1DDataset,
    folds,
    exp_idx: int,
    cfg: dict,
    exp_name: str,
    device: torch.device,
    tb_root: str,
    weight_root: str,
    Tmax: float,
):
    set_random_seed(cfg.get("seed", 777))

    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 0))
    total_steps = int(cfg.get("total_steps", 3000))
    eval_interval = int(cfg.get("eval_interval", 100))

    metric_names = params.get("metric_names", ["MAE", "MAPE"])

    # collocation 개수
    N_f = int(cfg.get("N_f", 2048))
    N_ic = int(cfg.get("N_ic", 256))
    N_bc_front = int(cfg.get("N_bc_front", 256))
    N_bc_back = int(cfg.get("N_bc_back", 256))

    # loss weight
    w_pde = float(cfg.get("w_pde", 1.0))
    w_ic = float(cfg.get("w_ic", 1.0))
    w_bc = float(cfg.get("w_bc", 1.0))
    w_data = float(cfg.get("w_data", 1.0))

    # pulse
    q0 = float(cfg.get("q0", 5e4))
    t_pulse_start = float(cfg.get("t_pulse_start", 0.0))
    t_pulse_end = float(cfg.get("t_pulse_end", 0.1))

    def q_pulse_fn(t: torch.Tensor) -> torch.Tensor:
        return q_pulse_square(t, q0=q0, t_start=t_pulse_start, t_end=t_pulse_end)

    kfold_n = len(folds)
    kfold_best_metrics = []
    kfold_last_metrics = []

    for kfold_idx, (train_idx, valid_idx) in enumerate(folds):
        # TensorBoard
        tb_log_dir = os.path.join(
            tb_root,
            f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_exp{exp_idx:03d}_fold{kfold_idx+1:02d}",
        )
        writer = SummaryWriter(log_dir=tb_log_dir)

        # dataset split
        train_subset = Subset(dataset, train_idx)
        valid_subset = Subset(dataset, valid_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        print(f"EXP {exp_idx:03d} | Fold {kfold_idx+1}/{kfold_n}")
        print(f"Train size: {len(train_subset)} | Valid size: {len(valid_subset)}")

        # 모델, optimizer
        model, optimizer, model_dim = build_model_from_params(params, cfg, device)
        Lz = float(model.Lz)
        Lx = float(cfg.get("Lx", 0.01))  # 2D용
        print(f"Model dimension: {model_dim}D | Lz: {Lz:.6f} m | Lx: {Lx:.6f} m")

        # train 로그 누적
        train_data_loss_list = []
        train_metric_lists = [[] for _ in metric_names]

        best_metric = float("inf")
        last_main_metric = None
        t_start = time.time()
        step_count = 0

        train_iter = iter(train_loader)

        while step_count < total_steps:
            step_count += 1
            model.train()
            optimizer.zero_grad()

            # ----- PDE residual -----
            if model_dim == 1:
                z_f = torch.rand(N_f, 1, device=device) * Lz
                t_f = torch.rand(N_f, 1, device=device) * Tmax
                x_f = torch.cat([z_f, t_f], dim=1)  # (N_f, 2)
            else:  # 2D
                x_f_x = torch.rand(N_f, 1, device=device) * Lx
                z_f = torch.rand(N_f, 1, device=device) * Lz
                t_f = torch.rand(N_f, 1, device=device) * Tmax
                x_f = torch.cat([x_f_x, z_f, t_f], dim=1)  # (N_f, 3)

            r_pde = model.pde_residual(x_f)
            loss_pde = torch.mean(r_pde ** 2)

            loss_pde = torch.mean(r_pde ** 2)

            # ----- Initial condition (t=0) -----
            if model_dim == 1:
                z_ic = torch.rand(N_ic, 1, device=device) * Lz
                t_ic = torch.zeros_like(z_ic, device=device)
                x_ic = torch.cat([z_ic, t_ic], dim=1)
            else:
                x_ic_x = torch.rand(N_ic, 1, device=device) * Lx
                z_ic = torch.rand(N_ic, 1, device=device) * Lz
                t_ic = torch.zeros_like(z_ic, device=device)
                x_ic = torch.cat([x_ic_x, z_ic, t_ic], dim=1)

            r_ic = model.ic_residual(x_ic)
            loss_ic = torch.mean(r_ic ** 2)


            # ----- Front BC (z=0) -----
            if model_dim == 1:
                z_bc_front = torch.zeros(N_bc_front, 1, device=device)
                t_bc_front = torch.rand(N_bc_front, 1, device=device) * Tmax
                x_bc_front = torch.cat([z_bc_front, t_bc_front], dim=1)
            else:
                x_bc_front_x = torch.rand(N_bc_front, 1, device=device) * Lx
                z_bc_front = torch.zeros(N_bc_front, 1, device=device)
                t_bc_front = torch.rand(N_bc_front, 1, device=device) * Tmax
                x_bc_front = torch.cat([x_bc_front_x, z_bc_front, t_bc_front], dim=1)

            r_bc_front = model.bc_front_residual(x_bc_front, q_pulse_fn)
            loss_bc_front = torch.mean(r_bc_front ** 2)


            # ----- Back BC (z=Lz) -----
            if model_dim == 1:
                z_bc_back = torch.full((N_bc_back, 1), fill_value=Lz, device=device)
                t_bc_back = torch.rand(N_bc_back, 1, device=device) * Tmax
                x_bc_back = torch.cat([z_bc_back, t_bc_back], dim=1)
            else:
                x_bc_back_x = torch.rand(N_bc_back, 1, device=device) * Lx
                z_bc_back = torch.full((N_bc_back, 1), fill_value=Lz, device=device)
                t_bc_back = torch.rand(N_bc_back, 1, device=device) * Tmax
                x_bc_back = torch.cat([x_bc_back_x, z_bc_back, t_bc_back], dim=1)

            r_bc_back = model.bc_back_residual(x_bc_back)
            loss_bc_back = torch.mean(r_bc_back ** 2)


            loss_bc = loss_bc_front + loss_bc_back

            # ----- Data loss -----
            try:
                x_data, T_data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_data, T_data = next(train_iter)

            x_data = x_data.to(device)
            T_data = T_data.to(device)

            # 1D / 2D에 따라 입력 형식 맞추기
            if model_dim == 1:
                # 1D 모델: Dataset이 이미 [z, t] 형태이므로 그대로 사용
                x_for_model = x_data  # (B, 2)
            else:
                # 2D 모델: Dataset은 [z, t]만 있으므로,
                # x=0을 앞에 붙여 [x, z, t] 형태로 확장
                z = x_data[:, 0:1]              # (B, 1)
                t = x_data[:, 1:2]              # (B, 1)
                x_coord = torch.zeros_like(z)   # (B, 1), x=0
                x_for_model = torch.cat([x_coord, z, t], dim=1)  # (B, 3)

            loss_data = model.data_loss(x_for_model, T_data)

            # 총 손실
            loss = (
                w_pde * loss_pde
                + w_ic * loss_ic
                + w_bc * loss_bc
                + w_data * loss_data
            )
            loss.backward()
            optimizer.step()

            # train 데이터 손실/metric 저장
            train_data_loss_list.append(loss_data.item())

            with torch.no_grad():
                T_pred_batch = model(x_for_model)

                # Kelvin -> Celsius 변환 (metric에서만 사용)
                T_data_C = T_data - 273.15
                T_pred_C = T_pred_batch - 273.15

                for idx, mname in enumerate(metric_names):
                    if mname.upper() == "MAE":
                        # MAE는 Kelvin/섭씨 동일하지만, 명시적으로 섭씨 기준으로 계산
                        v = torch.mean(torch.abs(T_pred_C - T_data_C)).item()
                    elif mname.upper() == "MAPE":
                        eps = 1e-8
                        v = torch.mean(
                            torch.abs((T_data_C - T_pred_C) / (T_data_C + eps))
                        ).item()
                    else:
                        raise NotImplementedError(f"Unsupported metric: {mname}")
                    train_metric_lists[idx].append(v)


            # ----- eval -----
            if (
                step_count % eval_interval == 0
                or step_count == 1
                or step_count == total_steps
            ):
                train_data_loss_mean = float(np.mean(train_data_loss_list)) if train_data_loss_list else 0.0
                train_metric_mean = [
                    float(np.mean(lst)) if lst else 0.0 for lst in train_metric_lists
                ]

                # Validation
                model.eval()
                val_data_loss_list = []
                val_metric_lists = [[] for _ in metric_names]

                with torch.no_grad():
                    for x_val, T_val in valid_loader:
                        x_val = x_val.to(device)
                        T_val = T_val.to(device)

                        # 1D / 2D에 따라 입력 형식 맞추기
                        if model_dim == 1:
                            x_for_model = x_val  # (B, 2) = [z, t]
                        else:
                            z = x_val[:, 0:1]
                            t = x_val[:, 1:2]
                            x_coord = torch.zeros_like(z)
                            x_for_model = torch.cat([x_coord, z, t], dim=1)  # (B, 3) = [x, z, t]

                        T_pred_val = model(x_for_model)
                        v_loss = model.data_loss(x_for_model, T_val)
                        val_data_loss_list.append(v_loss.item())

                        # Kelvin -> Celsius 변환 (metric에서만 사용)
                        T_val_C = T_val - 273.15
                        T_pred_val_C = T_pred_val - 273.15

                        for idx, mname in enumerate(metric_names):
                            if mname.upper() == "MAE":
                                v = torch.mean(torch.abs(T_pred_val_C - T_val_C)).item()
                            elif mname.upper() == "MAPE":
                                eps = 1e-8
                                v = torch.mean(
                                    torch.abs((T_val_C - T_pred_val_C) / (T_val_C + eps))
                                ).item()
                            else:
                                raise NotImplementedError(f"Unsupported metric: {mname}")
                            val_metric_lists[idx].append(v)



                val_data_loss_mean = float(np.mean(val_data_loss_list)) if val_data_loss_list else 0.0
                val_metric_mean = [
                    float(np.mean(lst)) if lst else 0.0 for lst in val_metric_lists
                ]

                # 주요 metric: 첫 번째 metric (없으면 val_data_loss)
                if metric_names:
                    main_val_metric = val_metric_mean[0]
                else:
                    main_val_metric = val_data_loss_mean
                last_main_metric = main_val_metric

                # best 갱신 시 모델 저장
                if main_val_metric < best_metric:
                    best_metric = main_val_metric
                    ckpt_path = os.path.join(
                        weight_root,
                        f"best_exp{exp_idx:03d}_fold{kfold_idx+1:02d}.pt",
                    )
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "best_metric": best_metric,
                            "step": step_count,
                            "fold_idx": kfold_idx,
                            "hparams": {
                                key: params[key]
                                for key in all_search_fields
                            }
                            if all_search_fields
                            else params,
                        },
                        ckpt_path,
                    )

                # TensorBoard
                writer.add_scalar("Loss/train_data", train_data_loss_mean, step_count)
                writer.add_scalar("Loss/val_data", val_data_loss_mean, step_count)
                for idx, mname in enumerate(metric_names):
                    writer.add_scalar(
                        f"Metric/train_{mname}", train_metric_mean[idx], step_count
                    )
                    writer.add_scalar(
                        f"Metric/val_{mname}", val_metric_mean[idx], step_count
                    )
                writer.add_scalar("Defect/z0", model.z0.item(), step_count)
                writer.add_scalar("Defect/z1", model.z1.item(), step_count)

                lr_now = optimizer.param_groups[0]["lr"]
                writer.add_scalar("LR", lr_now, step_count)

                # 콘솔 출력
                train_metric_str = " | ".join(
                    f"{name}(Train): {val:.4f}"
                    for name, val in zip(metric_names, train_metric_mean)
                )
                val_metric_str = " | ".join(
                    f"{name}(Val): {val:.4f}"
                    for name, val in zip(metric_names, val_metric_mean)
                )

                elapsed = time.time() - t_start
                print(
                    f"EXP {exp_idx:03d} | Fold {kfold_idx+1:02d} | Step {step_count:05d} | "
                    f"LR: {lr_now:.6e} | Train Loss(data): {train_data_loss_mean:.4e} | "
                    f"{train_metric_str} || Valid Loss(data): {val_data_loss_mean:.4e} | "
                    f"{val_metric_str} | Best(main metric): {best_metric:.4f} | "
                    f"z0: {model.z0.item():.4f}, z1: {model.z1.item():.4f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

                # reset train stats
                train_data_loss_list = []
                train_metric_lists = [[] for _ in metric_names]

        t_end = time.time()
        print(
            f"EXP {exp_idx:03d} | Fold {kfold_idx+1:02d} | "
            f"Best Metric: {best_metric:.4f} | Elapsed: {t_end - t_start:.2f} s"
        )
        kfold_best_metrics.append(best_metric)
        kfold_last_metrics.append(last_main_metric if last_main_metric is not None else best_metric)

        writer.close()

    print(
        f"EXP {exp_idx:03d} | "
        f"Total Best Metric Mean: {np.mean(kfold_best_metrics):.4f} | "
        f"Best Metric Std: {np.std(kfold_best_metrics):.4f} | "
        f"Total Last Metric Mean: {np.mean(kfold_last_metrics):.4f} | "
        f"Last Metric Std: {np.std(kfold_last_metrics):.4f}"
    )
    print(f"All Best Metrics: {kfold_best_metrics}")
    print(f"All Last Metrics: {kfold_last_metrics}")

    return float(np.mean(kfold_best_metrics)), float(np.mean(kfold_last_metrics))


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for 1D AirGap PINN (CSV surface data)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument("--name", type=str, required=True, help="Experiment name.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # GPU 설정
    gpu_id = int(cfg.get("gpu_id", 0))
    if torch.cuda.is_available() and gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    # 실험 이름
    exp_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.name}"

    # 디렉토리 및 로거 설정
    log_dir, weight_dir, tb_dir = setup_directories(cfg, exp_name)

    # config 백업
    shutil.copy(args.config, os.path.join(weight_dir, "config.yaml"))

    # seed
    set_random_seed(cfg.get("seed", 777))

    # Dataset (CSV)
    csv_path = cfg["csv_path"]  # 절대 경로 또는 상대 경로
    dataset = SurfaceTemp1DDataset(csv_path)
    n_samples = len(dataset)

    # Tmax (time domain 상한)
    if "Tmax" in cfg:
        Tmax = float(cfg["Tmax"])
    else:
        # dataset에서 time 최대값 사용
        Tmax = float(dataset.x[:, 1].max().item())

    print(f"Total samples: {n_samples}, Tmax: {Tmax:.6f}")

    # K-fold
    kfold_n = int(cfg.get("kfold_n", 5))
    folds = make_kfold_indices(n_samples, kfold_n)
    print(f"K-fold: {kfold_n}")
    for i, (train_idx, valid_idx) in enumerate(folds):
        print(f"===== Fold {i+1} =====")
        print(f"Train size: {len(train_idx)}, Valid size: {len(valid_idx)}")

    # 하이퍼파라미터 조합
    hparams = cfg.get("hparams", {})
    all_search_fields = [k[:-7] for k in hparams.keys() if k.endswith("_search")]
    all_combinations = get_hyperparameter_combinations(hparams)

    print(f"Search fields: {all_search_fields}")
    print(f"Total number of experiments: {len(all_combinations)}")

    global_best_metric = float("inf")

    for idx, params in enumerate(all_combinations, start=1):
        print("\n" + "=" * 60)
        print(f"Experiment {idx}/{len(all_combinations)}")
        print("=" * 60)

        if all_search_fields:
            print("Hyperparameters (search fields):")
            for key in all_search_fields:
                print(f"  {key}: {params[key]}")
        else:
            print("Hyperparameters (fixed/default):")
            for key, val in params.items():
                print(f"  {key}: {val}")
        print("=" * 60)

        best_metric, last_metric = run_experiment(
            params=params,
            all_search_fields=all_search_fields,
            dataset=dataset,
            folds=folds,
            exp_idx=idx,
            cfg=cfg,
            exp_name=exp_name,
            device=device,
            tb_root=tb_dir,
            weight_root=weight_dir,
            Tmax=Tmax,
        )

        if best_metric < global_best_metric:
            global_best_metric = best_metric
            print(f"[GLOBAL] New best metric: {global_best_metric:.4f}")

    print(
        f"[GLOBAL] Final best metric over all experiments: {global_best_metric:.4f}"
    )


if __name__ == "__main__":
    main()
