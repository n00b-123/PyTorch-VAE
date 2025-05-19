import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.strategies import DDPStrategy

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()

# ===== Config loading =====
print(f"[INFO] Loading config from {args.filename}")
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
        print("[INFO] Config file loaded successfully.")
    except yaml.YAMLError as exc:
        print("[ERROR] Failed to load config file:")
        print(exc)
        exit(1)

# ===== TensorBoard Logger =====
print(f"[INFO] Initializing TensorBoardLogger at {config['logging_params']['save_dir']}")
tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['model_params']['name'],
)

# ===== Seeding =====
print(f"[INFO] Seeding with {config['exp_params']['manual_seed']}")
seed_everything(config['exp_params']['manual_seed'], True)

# ===== Model Setup =====
print(f"[INFO] Initializing model: {config['model_params']['name']}")
model = vae_models[config['model_params']['name']](**config['model_params'])

# ===== Experiment Wrapper =====
experiment = VAEXperiment(model, config['exp_params'])

# ===== Dataset Setup =====
print("[INFO] Initializing dataset...")
data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()

# ===== Verifying data path =====
data_path = config["data_params"].get("data_path", "N/A")
print(f"[INFO] Checking if data path exists: {data_path}")
if not os.path.exists(data_path):
    print(f"[ERROR] Data path '{data_path}' not found.")
    exit(1)
else:
    print(f"[INFO] Data found at '{data_path}'")

# ===== Trainer Setup =====
print("[INFO] Setting up Trainer...")
runner = Trainer(
    logger=tb_logger,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            monitor="val_loss",
            save_last=True,
        ),
    ],
    strategy=DDPStrategy(find_unused_parameters=False),
    **config['trainer_params']
)

# ===== Creating Directories =====
print("[INFO] Creating directories for samples and reconstructions...")
Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

# ===== Start Training =====
print(f"\n======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
