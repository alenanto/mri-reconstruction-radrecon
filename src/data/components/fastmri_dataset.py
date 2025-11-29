import csv
from pathlib import Path

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset
from src.models.components.varnet import sens_expand,sens_reduce
from src.models.components import mri
import matplotlib.pyplot as plt
from src.data.transforms import apply_ssdu_mask
from src.utils import c2r, create_radial_traj


def read_slice_indices(csv_path: str) -> dict[str, int]:
    """Read slice indices from csv file.

    Args:
        csv_path: Path to the csv file.

    Returns:
        Dictionary containing slice index for each case.
    """
    slice_indices = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slice_indices[row['case']] = int(row['slice_index'])
    return slice_indices


class FastMRIBreastDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        time_frames: int = 22,
        spokes_per_frame: int = 13,
        base_res: int = 320,
        center_partition: int = 31,
        images_per_slab: int = 192,
        est_sens_maps: bool = False,
        ssdu: bool = False,
    ):
        """fastMRI breast dataset object.

        Args:
            dataset_path: Path to the subject files.
            time_frames: Number of time frames for radially undersampling.
                Defaults to 22.
            spokes_per_frame: Number of spokes per frame for radially
                undersampling. Defaults to 13.
            base_res: Base resolution (e.g. samples along x-axis).
                Defaults to 320.
            center_partition: The center partition index. Defaults to 31.
            images_per_slab: Total number of images per slab. Defaults to 192.
            est_sens_maps: Whether to use estimated coil sensitivity maps.
                Defaults to False.
            ssdu: Whether to apply spoke-wise undersampling mask to perform
                SSDU.
        """
        # Select the preprocssed subject slices and its estimated csm
        self.subject_path_list = sorted(
            Path(dataset_path).rglob('fastMRI_breast_*_1_*.h5')
        )
        self.time_frames = time_frames
        self.spokes_per_frame = spokes_per_frame
        self.base_res = base_res
        self.center_partition = center_partition
        self.images_per_slab = images_per_slab
        self.est_sens_maps = est_sens_maps
        self.ssdu = ssdu

        self.traj = create_radial_traj(
            time_frames, spokes_per_frame, base_res, gind=1
        )
        # Rotate image 180° to fix orientation
        self.traj[..., :] *= -1

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get data sample at given slice index.

        Args:
            index: Slice index.

        Returns:
            Either the radially undersampled k-space data, ground truth image,
            coil sensitivity maps and trajectory or the theta data, lambda
            data, coil sensitivity maps, trajectory, theta mask and ground
            truth image (if SSDU flag was set to `True`).
        """
        subject_path = Path(self.subject_path_list[index])

        # Load k-space and coil sensitivity map from HDF5 file
        with h5.File(subject_path, 'r') as f:
            ksp = f['kspace'][()]  # Shape: (16, 288, 640)
            csm = f['coil_sens'][()].astype(
                np.complex64
            )  # Shape: (16, 320, 320)
            
            gt=f['gt'][()] # Shape: (320, 320)

            # Analog to ksp, rotate csm 180° to fix orientation
            
            gt=np.flip(gt, axis=(0, 1)).copy()

            # Analog to ksp, rotate csm 180° to fix orientation
            csm = np.flip(csm, axis=(1, 2)).copy()

        # Perform sorting of spokes to get time frames
        N_spokes_prep = self.time_frames * self.spokes_per_frame
        ksp_redu = ksp[
            :, :N_spokes_prep, :
        ]  # Reduce to be int multiple of spokes

        ksp_sorted = np.swapaxes(ksp_redu, 0, 1)
        ksp_sorted = np.reshape(
            ksp_sorted,
            [self.time_frames, self.spokes_per_frame]
            + list(ksp_sorted.shape[1:]),
        )  # Shape: (22, 13, 16, 640)
        ksp_sorted = np.swapaxes(ksp_sorted, 1, 2)  # Shape: (22, 16, 13, 640)
        
    
    

        if self.ssdu:
            # Apply SSDU mask
            theta_data, lambda_data, theta_traj,traj_lambda, theta_mask = apply_ssdu_mask(
                ksp_sorted[:, :, ::6, :],
            self.traj[:, ::6, :, :],
            )
            return (
                torch.from_numpy(c2r(theta_data)),
                torch.from_numpy(c2r(lambda_data)),
                torch.from_numpy(csm),  # (16, 320, 320)
                torch.from_numpy(theta_traj),
                torch.from_numpy(traj_lambda),
                torch.from_numpy(theta_mask),
                torch.from_numpy(self.traj[:, ::6, :, :]),  # (22, 13, 640, 2)
                torch.from_numpy(gt),
                torch.from_numpy(ksp_sorted[:, :, ::6, :]),
        
            )

        return (
            torch.from_numpy(c2r(ksp_sorted[:, :, ::6, :])),  # (22, 16, 13, 640, 2)
            torch.from_numpy(c2r(ksp_sorted[:, :, ::6, :])),
            torch.from_numpy(csm),  # (16, 320, 320)
            torch.from_numpy(self.traj[:, ::6, :, :]),
            torch.from_numpy(self.traj[:, ::6, :, :]),
            torch.zeros(ksp_sorted[:, :, ::6, :].shape[2], dtype=bool),
            torch.from_numpy(self.traj[:, ::6, :, :]),  # (22, 13, 640, 2)
            torch.from_numpy(gt),
            torch.from_numpy(ksp_sorted[:, :, ::6, :]),
        )

    def __len__(self) -> int:
        """Returns the length of the DCE dataset.

        Returns:
            Length of the DCE dataset.
        """
        return len(self.subject_path_list)
