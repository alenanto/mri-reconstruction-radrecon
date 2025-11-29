import numpy as np
import numpy.typing as npt
import sigpy as sp
from sigpy import linop
from sigpy.mri import app


def create_radial_traj(
    num_time_frames: int,
    num_spokes_per_frame: int,
    base_res: int,
    gind: int = 2,
) -> npt.ArrayLike:
    """Create a radial trajectory / undersampling mask.

    Args:
        num_time_frames: Number of time frames/points.
        num_spokes_per_frame: Number of spokes per frame.
        base_res: Base resolution (e.g. samples along x-axis).
        gind: Geometry index. Defaults to 2.

    Returns:
        Radial trajectory / undersampling mask.
    """
    num_tot_spokes = num_spokes_per_frame * num_time_frames
    num_samples_per_spoke = base_res * 2

    base_lin = np.arange(num_samples_per_spoke).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)

    base_rot = np.arange(num_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((num_tot_spokes, num_samples_per_spoke, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    return traj.reshape(
        num_time_frames, num_spokes_per_frame, num_samples_per_spoke, 2
    )


def undersample(
    traj_t: npt.ArrayLike,
    coil_sens: npt.ArrayLike,
    image: npt.ArrayLike,
    num_time_frames: int,
    num_coil: int,
    num_spokes_per_frame: int,
    num_samples_per_spoke: int,
) -> npt.ArrayLike:
    """Undersamples the given k-space data using a radial trajectory.

    Args:
        traj_t: Radial trajectory / undersampling mask.
        coil_sens: Coil sensitivity map.
        image: Image to be undersampled.
        num_time_frames: Number of time frames/points in the resulting undersampled k-space data.
        num_coil: Number of coil channels in the resulting undersampled k-space data.
        num_spokes_per_frame: Number of spokes per frame in the resulting undersampled k-space data.
        num_samples_per_spoke: Number of samples per spoke (base resolution) in the resulting undersampled k-space data.

    Returns:
        Radial undersampled k-space data.
    """
    radialx_ksp = np.zeros(
        [
            num_time_frames,
            num_coil,
            num_spokes_per_frame,
            num_samples_per_spoke,
        ],
        dtype=complex,
    )

    for n in range(0, num_time_frames):
        traj_r = traj_t[n, ...]
        F = linop.NUFFT(coil_sens.shape, traj_r)
        radialx_ksp[n, ...] = F * image[n, ...]

    return radialx_ksp


def est_coil_sens_maps(
    radial_ksp: npt.ArrayLike,
    traj_t: npt.ArrayLike,
    device=sp.Device(-1),
    show_pbar=False,
) -> npt.ArrayLike:
    """Coil sensitivity map estimation using ESPIRiT

    Args:
        radial_ksp: K-Space data of shape (time, coils, spokes, samples).
        traj_t: Radial trajectory of shape (time, spokes, samples, complex)
        device: Device for calculation. Defaults to sp.Device(-1).
        show_pbar: Show progress bar. Defaults to False.

    Returns:
        Estimated coil sensitivity maps of shape (c, x, y).
    """
    _, N_coils, _, N_samples = radial_ksp.shape
    base_res = N_samples // 2
    ishape = [N_coils] + [base_res] * 2
    t = 0  # Choose an arbitory time frame to estimate coil sens maps
    traj = traj_t[t]

    # Precompute density compensation
    dcf = (traj[..., 0] ** 2 + traj[..., 1] ** 2) ** 0.5

    # NUFFT operator for the current frame
    F = sp.linop.NUFFT(ishape, traj)

    # Apply density compensation and NUFFT adjoint
    cim = F.H(radial_ksp[t] * dcf)

    # Coil sensitivity map estimation using ESPIRiT
    ksp = sp.fft(cim, axes=(-2, -1))
    csm = app.EspiritCalib(
        ksp, crop=0, device=device, show_pbar=show_pbar
    ).run()

    return sp.to_device(csm)
