import numpy as np


def apply_ssdu_mask(
    input_data: np.ndarray,
    traj: np.ndarray,
    theta_lambda_ratio: int = 0.6,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split k-space measurements (omega) into two disjoint sets, theta and
    lambda.

    Args:
        input_data: Radial k-space data to apply mask to.
            Expected shape: (time, coil, spokes, samples).
        traj: Radial trajectory data. Expected shape: (time, spokes, samples).
        theta_lambda_ratio: Ratio of spokes for theta set to total spokes.
            Default is 0.75, meaning 75% of spokes will be used for theta
            and 25% for lambda.
        random_seed: Optional seed for reproducibility. Default is 42.

    Raises:
        ValueError: If input_data does not have shape (time, coil, spokes,
            samples)
        ValueError: If traj does not have shape (time, spokes, samples, 2

    Returns:
        K-space data for theta and lambda sets, trajectory for theta set and the
        theta mask.
    """
    if input_data.ndim != 4:
        raise ValueError('Data must have shape (time, coil, spokes, samples)')
    if traj.ndim != 4:
        raise ValueError(
            'Trajectory must have shape (time, spokes, samples, 2)'
        )

    # Extract the number of spokes from the input data
    num_spokes = input_data.shape[2]

    # Determine the number of theta spokes based on the ratio
    num_theta_spokes = int(round(theta_lambda_ratio * num_spokes))

    # Randomly shuffle the indices of the spokes
    np.random.seed(random_seed)
    spoke_indices = np.arange(num_spokes)
    np.random.shuffle(spoke_indices)

    # Select the first `num_theta_spokes` indices for theta set
    theta_indices = spoke_indices[:num_theta_spokes]

    # Create the theta mask
    theta_mask = np.zeros(num_spokes, dtype=bool)
    theta_mask[theta_indices] = True

    # Extract k-space data and trajectory for theta and lambda set
    kspace_theta = input_data[:, :, theta_mask, :]
    traj_theta = traj[:, theta_mask, :, :]
    kspace_lambda = input_data[:, :, ~theta_mask, :]
    traj_lambda = traj[:, ~theta_mask, :, :]

    return kspace_theta, kspace_lambda, traj_theta,traj_lambda, theta_mask
