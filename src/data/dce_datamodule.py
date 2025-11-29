"""
DCE DataModule for processing the dataset.

Copyright (c) Erik Goesche <erik.goesche@fau.de>.

Part of this code is licensed under the MIT License.
"""

from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.fastmri_dataset import FastMRIBreastDataset
from src.utils import (
    RankedLogger
)

log = RankedLogger(__name__, rank_zero_only=True)


class DCEDataModule(LightningDataModule):
    """`LightningDataModule` for working with DCE-MRI datasets."""

    def __init__(
        self,
        dataset_path: str = "data/fastMRI_breast/",
        train_val_test_split: tuple = (0.8, 0.1, 0.1),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        time_frames: int = 22,
        spokes_per_frame: int = 13,
        est_sens_maps: bool = False,
        ssdu: bool = False,
    ) -> None:
        """Initialize a `DCEDataModule`.

        Args:
            dataset_path: Path to the dataset directory.
                Defaults to `"data/fastMRI_breast/"`.
            train_val_test_split: The train, validation and test split.
                Defaults to `(55_000, 5_000, 10_000)`.
            batch_size: The batch size. Defaults to `1`.
            num_workers: The number of workers. Defaults to `0`.
            pin_memory: Whether to pin memory. Defaults to `False`.
            time_frames: The number of time frames for radially undersampling.
                Defaults to `22`.
            spokes_per_frame: The number of spokes per frame for radially
                undersampling. Defaults to `13`.
            est_sens_maps: Whether to use estimate coil sensitivity maps.
                Defaults to `False`.
            ssdu: Whether to use self-supervised learning by spoke-wise
                undersampling. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.dataset_name = str(Path(dataset_path)).split('/')[-1]
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        Returns:
            The number of dce classes (0).
        """
        return 0

    def prepare_data(self) -> None:
        """Prepare data if needed. Lightning ensures that `self.prepare_data()`
        is called only within a single process on CPU, so you can safely add
        your downloading logic within. In case of multi-node training, the
        execution of this hook depends upon `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

        This method is called by Lightning before `trainer.fit()`,
        `trainer.validate()`, `trainer.test()`, and `trainer.predict()`, so be
        careful not to execute things like random split twice! Also, it is
        called after `self.prepare_data()` and there is a barrier in between
        which ensures that all the processes proceed to `self.setup()` once the
        data is prepared and available for use.

        Args:
            stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`,
                or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # Load and split datasets only if not already loaded
        if not any([self.data_train, self.data_val, self.data_test]):
            dataset_params = {
                'dataset_path': self.hparams.dataset_path,
                'time_frames': self.hparams.time_frames,
                'spokes_per_frame': self.hparams.spokes_per_frame,
                'est_sens_maps': self.hparams.est_sens_maps,
                'ssdu': self.hparams.ssdu,
            }

            if self.dataset_name == 'fastMRI_breast':
                dataset = FastMRIBreastDataset(**dataset_params)
            else:
                raise ValueError(
                    f'Unknown dataset: {self.dataset_name}. Supported datasets: fastMRI_breast.'
                )

            self.data_train, self.data_val, self.data_test = random_split(
                dataset,
                self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        Returns:
            The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create and return the predict dataloader.

        Returns:
            The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`,
        `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        Args:
            stage: The stage being torn down. Either `"fit"`, `"validate"`,
                `"test"`, or `"predict"`. Defaults to ``None``.
        """
        pass

    def state_dict(self) -> dict:
        """Called when saving a checkpoint. Implement to generate and save the
        datamodule state.

        Returns:
            A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule
        state given datamodule `state_dict()`.

        Args:
            state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DCEDataModule()
