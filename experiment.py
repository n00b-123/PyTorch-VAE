import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = params.get('retain_first_backpass', False)
        self.log_dir = None  # Set after trainer is initialized

        self.training_losses = []  # Store per-epoch logs

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        # Save losses for custom logging
        self.training_losses.append({
            "loss": train_loss['loss'].item(),
            "Reconstruction_Loss": train_loss['Reconstruction_Loss'].item(),
            "KLD": train_loss['KLD'].item()
        })

        return train_loss['loss']

    def training_epoch_end(self, outputs):
        # Compute average losses
        if self.training_losses:
            avg_loss = sum(x['loss'] for x in self.training_losses) / len(self.training_losses)
            avg_recons = sum(x['Reconstruction_Loss'] for x in self.training_losses) / len(self.training_losses)
            avg_kld = sum(x['KLD'] for x in self.training_losses) / len(self.training_losses)

            # Get log dir
            if self.logger:
                log_dir = self.logger.log_dir
            else:
                log_dir = self.log_dir or "logs"

            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "loss_log.txt")
            with open(log_file, "a") as f:
                f.write(f"Epoch {self.current_epoch:03d} | "
                        f"Total Loss: {avg_loss:.4f} | "
                        f"Reconstruction Loss: {avg_recons:.4f} | "
                        f"KLD: {avg_kld:.4f}\n")

            # Reset epoch cache
            self.training_losses = []

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144, self.curr_device, labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        if self.params.get('LR_2') is not None:
            optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                    lr=self.params['LR_2'])
            optims.append(optimizer2)

        if self.params.get('scheduler_gamma') is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                         gamma=self.params['scheduler_gamma'])
            scheds.append(scheduler)

            if len(optims) > 1 and self.params.get('scheduler_gamma_2') is not None:
                scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                              gamma=self.params['scheduler_gamma_2'])
                scheds.append(scheduler2)
            return optims, scheds

        return optims
