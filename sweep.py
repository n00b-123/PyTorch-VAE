config_files = [
    "configs/sweeps/sweep_1_lr=0.001_kld=0.0001_wd=0.0001.yaml",
    "configs/sweeps/sweep_2_lr=0.001_kld=0.00025_wd=0.0.yaml",
    "configs/sweeps/sweep_3_lr=0.001_kld=0.00025_wd=0.0001.yaml",
    "configs/sweeps/sweep_4_lr=0.001_kld=0.001_wd=0.0.yaml",
    "configs/sweeps/sweep_5_lr=0.001_kld=0.001_wd=0.0001.yaml",
    "configs/sweeps/sweep_6_lr=0.005_kld=0.0001_wd=0.0.yaml",
    "configs/sweeps/sweep_7_lr=0.005_kld=0.0001_wd=0.0001.yaml",
    "configs/sweeps/sweep_8_lr=0.005_kld=0.00025_wd=0.0.yaml",
    "configs/sweeps/sweep_9_lr=0.005_kld=0.00025_wd=0.0001.yaml",
    "configs/sweeps/sweep_10_lr=0.005_kld=0.001_wd=0.0.yaml",
    "configs/sweeps/sweep_11_lr=0.005_kld=0.001_wd=0.0001.yaml",
    "configs/sweeps/sweep_12_lr=0.01_kld=0.0001_wd=0.0.yaml",
    "configs/sweeps/sweep_13_lr=0.01_kld=0.0001_wd=0.0001.yaml",
    "configs/sweeps/sweep_14_lr=0.01_kld=0.00025_wd=0.0.yaml",
    "configs/sweeps/sweep_15_lr=0.01_kld=0.00025_wd=0.0001.yaml",
    "configs/sweeps/sweep_16_lr=0.01_kld=0.001_wd=0.0.yaml",
    "configs/sweeps/sweep_17_lr=0.01_kld=0.001_wd=0.0001.yaml",
    "configs/sweeps/sweep_0_lr=0.001_kld=0.0001_wd=0.0.yaml"
]

import subprocess

for config in config_files:
    print(f"=== Running {config} ===")
    subprocess.run(["python", "run.py", "--config", config])
