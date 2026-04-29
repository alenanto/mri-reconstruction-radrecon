import argparse
from pathlib import Path

import h5py as h5
import numpy as np
import pandas as pd
import rootutils
import sigpy as sp
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.radial_imaging import create_radial_traj, est_coil_sens_maps


def load_slice_indices(xlsx_path: str) -> dict[str, int]:
    """Load slice indices from a XLSX file.

    Args:
        xlsx_path: Path to the XLSX file containing slice indices.

    Returns:
        Dictionary containing slice index for each case.
    """
    # Load the Excel file
    df = pd.read_excel(Path(xlsx_path))

    # Process the 'Center Slice' column: convert to list of ints
    df['Center Slice of Each Lesion (in order)'] = df[
        'Center Slice of Each Lesion (in order)'
    ].apply(
        lambda x: (
            [int(i.strip()) for i in str(x).split(',') if i.strip().isdigit()]
            if pd.notnull(x)
            else []
        )
    )

    # Build initial dict from the DataFrame: Case Number -> first center slice
    case_to_slice = {
        f'{int(row["Case Number"]):03}': row[
            'Center Slice of Each Lesion (in order)'
        ][0]
        for _, row in df.iterrows()
        if row['Center Slice of Each Lesion (in order)']
    }

    # Fill in full dictionary from 001 to 244, using default 80 where necessary
    slice_indices = {
        f'{i:03}': case_to_slice.get(f'{i:03}', 80) for i in range(1, 245)
    }

    return slice_indices


def process_subject(
    subject_path: str,
    slice_idx: int,
    output_path: str,
    images_per_slab: int = 192,
    center_partition: int = 31,
    skip_existing: bool = False,
    device: int = -1,
) -> None:
    """Process a single subject and save the processed k-space slice and its
    coil sensitivity map.

    Args:
        subject_path: Path to the subject file.
        slice_idx: Index of the slice that contains the lesion.
        images_per_slab: Total number of images per slab. Defaults to 192.
        center_partition: The center partition index. Defaults to 31.
        skip_existing: Whether to skip processing if the output file already
            exists. Defaults to False.
        device: Device to use for processing. Defaults to -1.
    """
    subject_name = subject_path.stem
    output_path = Path(output_path) / f"{subject_name}_{slice_idx}.h5"

    if skip_existing and output_path.exists():
        print(f"Skipping {subject_name}, already processed.")
        return

    print(f"Processing {subject_name}...")

    with h5.File(subject_path, 'r') as f:
        # Get the dataset
        ksp_f = f['kspace'][:].T  # Shape: (83, 16, 640, 288, 2)
        img_data = f["temptv"][:] # Shape: (192, 4, 320, 320)
    ksp_f = np.transpose(ksp_f, (4, 3, 2, 1, 0))

    ksp = ksp_f[0] + 1j * ksp_f[1]
    ksp = np.transpose(ksp, (3, 2, 0, 1))  # Shape: (83, 16, 288, 640)

    # zero-fill the slice dimension
    partitions = ksp.shape[0]
    shift = int(images_per_slab / 2 - center_partition)
    ksp_zf = np.zeros_like(ksp, shape=[images_per_slab] + list(ksp.shape[1:]))
    ksp_zf[shift : shift + partitions, ...] = ksp
    ksp_zf = sp.fft(ksp_zf, axes=(0,))  # Shape: (192, 16, 288, 640)

    # Select slice that contains the lesion
    ksp_slice = ksp_zf[slice_idx]  # Shape: (16, 288, 640)
    img_slice = img_data[slice_idx]
    gt_image = np.mean(img_slice, axis=0)
    # Add time dimension
    ksp_slice_t = ksp_slice[np.newaxis, ...]  # Shape: (1, 16, 288, 640)

    # Get trajectory with combined spokes for one time frame
    combined_traj = create_radial_traj(
        1, ksp_slice_t.shape[2], ksp_slice_t.shape[3] // 2, 1
    )
    print(device)
    # Estimate coil sensitivity maps to avoid inverse crime
    coil_sens = est_coil_sens_maps(ksp_slice_t, combined_traj, device=device)

    with h5.File(output_path, 'w') as f:
        f.create_dataset('kspace', data=ksp_slice)
        f.create_dataset('coil_sens', data=coil_sens)
        f.create_dataset('gt',data=gt_image)


def main(args):
    """Main function to process all subjects in the dataset.

    Args:
        args: Command-line arguments.
    """
    slice_indices_path = args.slice_indices_path
    data_dir = Path(args.dataset_path)

    device = sp.Device(0 if torch.cuda.is_available() else -1)

    # Load slice indices
    slice_indices = load_slice_indices(slice_indices_path)
    # Change each key to match the subject naming convention
    slice_indices = {
        f'fastMRI_breast_{int(k):03}_1': v for k, v in slice_indices.items()
    }

    # Get subject file paths
    subject_path_list = sorted(data_dir.rglob('fastMRI_breast_*_1.h5'))
    subject_path_list = subject_path_list[:200]  # Limit to first 100 subjects
    
    # Process each subject
    for subject_path in subject_path_list:
        subject_name = subject_path.stem
        if subject_name in slice_indices:
            process_subject(
                subject_path,
                slice_indices[subject_name],
                output_path=args.output_path,
                skip_existing=args.skip_existing,
                device=device,
            )
        else:
            print(
                f"Warning: No slice index found for {subject_name}, skipping."
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script reads all fastMRI breast subjects and
        processes the k-space data for the slice containing the lesion. The
        extracted slice and its estimated coil sensitivity map is saved as a 
        new HDF5 file with the same name as the original subject file but with 
        the slice index appended to the name.'''
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='/home/hpc/iwbi/iwbi101h/Rad_Recon/data/fastMRI_org',
        help='Path to the fastMRI breast dataset directory.',
    )
    parser.add_argument(
        '--slice-indices-path',
        type=str,
        default='/home/hpc/iwbi/iwbi101h/Rad_Recon/data/fastMRI_breast/slice_indices.xlsx',
        help='Path to the XLSX file containing slice indices.',
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/home/hpc/iwbi/iwbi101h/Rad_Recon/data/fastMRI_breast',
        help='Path to save the processed files.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help="Skip processing if the output file already exists.",
    )

    args = parser.parse_args()
    main(args)
