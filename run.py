import os
import yaml
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

from models import *
from experiment import VAEXperiment
from dataset import VAEDataset

import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.plugins import DDPPlugin

from ignite.engine import Engine, Events
from ignite.metrics import FID, InceptionScore

# === FID & IS Callback ===
class FIDISCallback(Callback):
    def __init__(self, every_n_epochs=10, latent_dim=128, num_samples=1024, log_file="fid_is_log.txt"):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.log_file = log_file

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        print(f"[INFO] Running FID/IS evaluation at epoch {epoch}...")

        vae = pl_module.model.eval()
        device = vae.decoder_input.weight.device

        val_loader = trainer.datamodule.test_dataloader()
        z = torch.randn(self.num_samples, self.latent_dim, device=device)
        fake = vae.decode(z)

        def resize_and_rescale(images):
            if images.min() < 0 or images.max() > 1:
                images = (images + 1) / 2
            return F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        fake = resize_and_rescale(fake)

        real, _ = next(iter(val_loader))
        real = resize_and_rescale(real.to(device))

        def evaluation_step(engine, batch):
            return fake, real

        evaluator = Engine(evaluation_step)
        FID(device=device).attach(evaluator, "fid")
        InceptionScore(device=device).attach(evaluator, "is")

        evaluator.run([None])  # one iteration

        fid_score = evaluator.state.metrics['fid']
        is_mean, is_std = evaluator.state.metrics['is']
        print(f"📊 Epoch {epoch} - FID: {fid_score:.4f} | IS: {is_mean:.4f} ± {is_std:.4f}")

        with open(os.path.join(trainer.logger.log_dir, self.log_file), "a") as f:
            f.write(f"Epoch {epoch} | FID: {fid_score:.4f} | IS: {is_mean:.4f} ± {is_std:.4f}\n")


# === CLI & Config ===
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c', dest="filename", metavar='FILE', help='path to the config file', default='configs/vae.yaml')
args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# === Logger, Seed, Paths ===
tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'])

seed_everything(config['exp_params']['manual_seed'], True)

# === Model & Data ===
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()

# === Trainer ===
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
        FIDISCallback(
            every_n_epochs=10,
            latent_dim=config["model_params"]["latent_dim"],
            num_samples=1024
        ),
    ],
    strategy=DDPPlugin(find_unused_parameters=False),
    **config['trainer_params']
)

# === Setup output folders ===
Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
