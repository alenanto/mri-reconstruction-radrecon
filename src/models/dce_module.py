"""
DCE Module for training the DCE model.

Copyright (c) Erik Goesche <erik.goesche@fau.de>.

Part of this code is licensed under the MIT License.
"""

from pathlib import Path

import h5py
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from src.models.components.varnet import sens_reduce
from src.models.components import mri
from src.utils import c2r, r2c
import matplotlib.pyplot as plt
import numpy as np


class DCELitModule(LightningModule):
    """LightningModule for the DCE reconstruction."""

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        output_dir: str = None,
    ) -> None:
        """Initialize a `DCELitModule`.

        Args:
            net: The model to train.
            criterion: The loss function.
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler to use for training.
            compile: Whether to compile the model.
            output_dir: The output directory. Defaults to `None`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = criterion

        self.output_dir = output_dir

        # metric objects for calculating and averaging values across batches
        self.train_psnr = PeakSignalNoiseRatio()
        self.val_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()

        self.train_ssim = StructuralSimilarityIndexMeasure()
        self.train_ssim2 = StructuralSimilarityIndexMeasure()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_ssim = StructuralSimilarityIndexMeasure()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation metric value
        self.val_psnr_best = MaxMetric()

    def forward(
        self, x: torch.Tensor, csm: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        Args:
            x: A tensor of images.
            csm: A tensor of coil sensitivity maps.
            mask: A tensor of masks.
        """
        return self.net(x, csm, mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before
        # training starts, so it's worth to make sure validation metrics don't
        # store results from these checks
        self.val_loss.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()
        self.val_psnr_best.reset()

    def model_step(
        self,
        batch: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],mode='train'
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Perform a single model step on a batch of data.

        Args:
            batch: A tuple containing:
                - u_kspace: undersampled k-space data of 
                    shape (batch, 2, time_frames, coils, spokes, samples).
                - target: Target labels (k-space in self-supervised, image in supervised).
                - csm: Coil sensitivity maps.
                - traj: Sampling trajectory.
                - theta_mask (optional, used in self-supervised).
                - omega_traj (optional, used in self-supervised).
                - gt_image (optional, only in batch of size 7, for evaluation).

        Returns:
            - loss: Computed loss.
            - logits: Model predictions.
            - gt_image: Ground truth image (only if available, otherwise None).
        """
        # Unpack always present components
         # === Unpack batch ===
        # === Unpack batch ===
        u_kspace, target, csm, traj_theta, traj_lambda = batch[:5]
        theta_masks, traj = batch[5:7]
        time_frames = u_kspace.shape[2]

        # === Forward passes ===
        logits_theta,dc_t = self.forward(u_kspace, csm, traj_theta)
        logits_lambda,dc_l = self.forward(target, csm, traj_lambda)
        
        self.log(
            "dc_t",
            dc_t,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "dc_l",
            dc_l,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # === TRAIN MODE ===
        if mode == "train":

            # --- Compute k-space predictions ---
            A = mri.SenseOp(traj[0], csm[0], device=u_kspace.device)
            logits_cc = r2c(logits_theta, axis=1).permute(1, 0, 2, 3) * csm
            kspace_theta = c2r(A.fwd(logits_cc))[None, :]

            logits_dd = r2c(logits_lambda, axis=1).permute(1, 0, 2, 3) * csm
            kspace_lambda = c2r(A.fwd(logits_dd))[None, :]

            # --- Prepare masks ---
            theta_mask = theta_masks[:, None, None, None, :, None].expand_as(kspace_theta)
            masked_theta = kspace_theta[~theta_mask].view(1, 2, time_frames, 16, -1, 640)
            masked_lambda = kspace_lambda[theta_mask].view(1, 2, time_frames, 16, -1, 640)
            kspace_gt = batch[8]

            def combine_kspace(kspace, u_known, u_pred, mask):
                
                # Match u_known to u_pred's dtype and device
                u_known = u_known.to(dtype=u_pred.dtype, device=u_pred.device)

                # Create zero tensor with same dtype/device as u_pred
                combined = torch.zeros_like(kspace, dtype=u_pred.dtype, device=u_pred.device)

                # Fill known and predicted regions
                combined[..., mask[0], :] = u_known
                combined[..., ~mask[0], :] = u_pred
                return combined

            # === Combine θ-known / λ-predicted and λ-known / θ-predicted ===
            kspace_full_theta = combine_kspace(kspace_theta,u_kspace, masked_theta, theta_masks)
            kspace_full_lambda = combine_kspace(kspace_lambda,target, masked_lambda, ~theta_masks)

            # === Helper inline function: reconstruct image from k-space ===
            def reconstruct(kspace):
                A2 = mri.SenseOp(traj[0], csm[0], dcf=True, device=u_kspace.device)
                return sens_reduce(r2c(kspace.squeeze(0)), csm, A2)

            gt_img = reconstruct(c2r(kspace_gt.squeeze(0)).unsqueeze(0))
            img_theta = reconstruct(kspace_full_theta)
            img_lambda = reconstruct(kspace_full_lambda)
            
            
            # === Helper inline function: compute SSIM + L1 metrics ===
            def compute_metrics(pred_img, gt_img):
                max_val = gt_img.abs().max()
                pred_norm = pred_img.abs() / (max_val + 1e-8)
                gt_norm = gt_img.abs() / (max_val + 1e-8)

                ssim_val = self.train_ssim2(pred_norm.unsqueeze(0), gt_norm.unsqueeze(0))
                l1_val = (
                    torch.linalg.vector_norm(pred_norm - gt_norm, ord=1)
                    / torch.linalg.vector_norm(gt_norm, ord=1)
                )
                return ssim_val, l1_val

            # === Compute metrics for θ and λ ===
            ssim_t, l1_t = compute_metrics(img_theta, gt_img)
            ssim_l, l1_l = compute_metrics(img_lambda, gt_img)

            # === Self-supervised k-space losses ===
            loss_theta = self.criterion(masked_theta, target)
            loss_lambda = self.criterion(masked_lambda, u_kspace)

            # === Combine losses ===
            loss_t = loss_theta + 0.1 * l1_t + 0.1 * (1 - ssim_t)
            loss_l = loss_lambda + 0.1 * l1_l + 0.1 * (1 - ssim_l)
            loss = loss_t + 0.5*loss_l
            
            self.log(
            "loss_t",
             loss_t,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
            self.log(
            "loss_l",
             loss_l,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
          

            gt_image = batch[7] if len(batch) == 9 else None
            return loss, logits_theta, gt_image

        # === TEST MODE ===
        if mode == "test":
            gt_image = batch[7] if len(batch) == 9 else None
            return None,logits_theta, gt_image

         

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training
        set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of
                images and target labels.
            batch_idx: The index of the current batch.

        Returns:
            A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch,mode='train')

        # update and log metrics
        self.train_loss(loss)
        
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        
        
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if targets is not None:
            #self.train_psnr(r2c(preds,axis=1), targets.unsqueeze(1))
            #self.train_ssim(torch.abs(r2c(preds,axis=1)),torch.abs(targets.unsqueeze(1)))
            
            pred = r2c(preds, axis=1).abs()
            tgts = targets.unsqueeze(1).abs()

            # Normalize both using target max value
            max_val = tgts.max()
            pred_norm = pred / (max_val + 1e-8)
            tgt_norm  = tgts / (max_val + 1e-8)


            # PSNR & SSIM
            self.train_psnr(pred_norm, tgt_norm)
            self.train_ssim(pred_norm, tgt_norm)
            
            self.log(
                "train/psnr",
                self.train_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "train/ssim",
                self.train_ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the
        validation set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of
                images and target labels.
            batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch,mode='train')

        # update and log metrics
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if targets is not None:
            #self.val_psnr(r2c(preds,axis=1), targets.unsqueeze(1))
            #self.val_ssim(torch.abs(r2c(preds,axis=1)),torch.abs(targets.unsqueeze(1)))
            
            pred = r2c(preds, axis=1).abs()
            tgts = targets.unsqueeze(1).abs()

            # Normalize both using target max value
            max_val = tgts.max()
            pred_norm = pred / (max_val + 1e-8)
            tgt_norm  = tgts / (max_val + 1e-8)


            # PSNR & SSIM
            self.val_psnr(pred_norm, tgt_norm)
            self.val_ssim(pred_norm, tgt_norm)
          

            
            self.log(
                "val/psnr",
                self.val_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "val/ssim",
                self.val_ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # Check if val psnr has been computed at least once
        if self.val_psnr.update_count > 0:
            psnr = self.val_psnr.compute()  # get current val psnr
            self.val_psnr_best(psnr)  # update best so far val psnr
            # log `val_psnr_best` as a value through `.compute()` method, instead
            # of as a metric object otherwise metric would be reset by lightning
            # after each epoch
            self.log(
                "val/psnr_best",
                self.val_psnr_best.compute(),
                sync_dist=True,
                prog_bar=True,
            )

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of
                images and target labels.
            batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch,mode='test')

        # update and log metrics
        

        if targets is not None:
            #self.test_psnr(r2c(preds,axis=1), targets.unsqueeze(1))
            #self.test_ssim(torch.abs(r2c(preds,axis=1)),torch.abs(targets.unsqueeze(1)))
            
            pred = r2c(preds, axis=1).abs()
            tgts = targets.unsqueeze(1).abs()

            # Normalize both using target max value
            max_val = tgts.max()
            pred_norm = pred / (max_val + 1e-8)
            tgt_norm  = tgts / (max_val + 1e-8)

            
            # PSNR & SSIM
            psnr_val = self.test_psnr(pred_norm, tgt_norm)
            ssim_val = self.test_ssim(pred_norm, tgt_norm)
            #mse_val=self.test_mse(pred_norm, tgt_norm)
            nmse_val = torch.sum((pred_norm - tgt_norm) ** 2) / (torch.sum(tgt_norm ** 2) + 1e-8)

            print(f"PSNR: {psnr_val.item():.4f}, SSIM: {ssim_val.item():.4f}, NMSE: {nmse_val:.6f}")
            
            self.log(
                "test/psnr",
                self.test_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/ssim",
                self.test_ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # Save preds with their corresponding gt without the batch dimension
        targets = targets.cpu().numpy().squeeze(0)
        preds = preds.cpu().numpy().squeeze(0)
        preds = abs(r2c(preds, axis=0))

        # Ensure the directory exists
        if self.output_dir is None:
            self.output_dir = '.'
        output_path = Path(self.output_dir) / 'preds'
        output_path.mkdir(parents=True, exist_ok=True)

        with h5py.File(
            output_path / f'recon_{str(batch_idx).zfill(3)}.h5', 'w'
        ) as f:
            f.create_dataset('recon', data=preds)
            f.create_dataset('gt', data=targets)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single prediction step on a batch of data from the test
            set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of
                images and target labels.
            batch_idx: The index of the current batch.
        """
        _, preds, _ = self.model_step(batch)

        # Save preds without the batch dimension
        preds = preds.cpu().numpy().squeeze(0)

        # Get magnitude of the predictions
        preds = abs(r2c(preds, axis=0))

        # Ensure the directory exists
        if self.output_dir is None:
            self.output_dir = '.'
        output_path = Path(self.output_dir) / 'preds'
        output_path.mkdir(parents=True, exist_ok=True)

        with h5py.File(
            output_path / f'recon_{str(batch_idx).zfill(3)}.h5', 'w'
        ) as f:
            f.create_dataset('recon', data=preds)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train +
        validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using
        DDP.

        Args:
            stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, any]:
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate
            schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DCELitModule(None, None, None, None)
