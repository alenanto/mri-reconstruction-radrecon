"""
Fourier and Sensitivity Encoding (SENSE) Operations.

Copyright (c) Zhengguo Tan <zhengguo.tan@gmail.com>.
Copyright (c) Erik Goesche <erik.goesche@fau.de>.

Part of this code is licensed under the MIT License.
"""

import numpy as np
import sigpy
import torch


def fftc(
    input: torch.Tensor | np.ndarray,
    axes: tuple | None = (-2, -1),
    norm: str | None = 'ortho',
) -> torch.Tensor | np.ndarray:
    """Centered Fast Fourier Transform (FFT) operation.

    Args:
        input: Input data with shape [..., N_y, N_x] or [..., N_y, N_x, 2].
        axes: Axes to perform the FFT operation. Defaults to (-2, -1).
        norm: Normalization flag for scaling the output. Defaults to 'ortho'.

    Returns:
        Output data with shape [..., N_y, N_x] or [..., N_y, N_x, 2].
    """
    if isinstance(input, np.ndarray):
        tmp = np.fft.ifftshift(input, axes=axes)
        tmp = np.fft.fftn(tmp, axes=axes, norm=norm)
        output = np.fft.fftshift(tmp, axes=axes)

    elif isinstance(input, torch.Tensor):
        tmp = torch.fft.ifftshift(input, dim=axes)
        tmp = torch.fft.fftn(tmp, dim=axes, norm=norm)
        output = torch.fft.fftshift(tmp, dim=axes)

    return output


def ifftc(
    input: torch.Tensor | np.ndarray,
    axes: tuple | None = (-2, -1),
    norm: str | None = 'ortho',
) -> torch.Tensor | np.ndarray:
    """Centered Inverse Fast Fourier Transform (IFFT) operation.

    Args:
        input: Input data with shape [..., N_y, N_x] or [..., N_y, N_x, 2].
        axes: Axes to perform the IFFT operation. Defaults to (-2, -1).
        norm: Normalization flag for scaling the output. Defaults to 'ortho'.

    Returns:
        Output data with shape [..., N_y, N_x] or [..., N_y, N_x, 2].
    """
    if isinstance(input, np.ndarray):
        tmp = np.fft.ifftshift(input, axes=axes)
        tmp = np.fft.ifftn(tmp, axes=axes, norm=norm)
        output = np.fft.fftshift(tmp, axes=axes)

    elif isinstance(input, torch.Tensor):
        tmp = torch.fft.ifftshift(input, dim=axes)
        tmp = torch.fft.ifftn(tmp, dim=axes, norm=norm)
        output = torch.fft.fftshift(tmp, dim=axes)

    return output


class SenseOp:
    """Sensitivity Encoding (SENSE) Operators

    Reference:
        * Pruessmann KP, Weiger M, Börnert P, Boesiger P.
          Advances in sensitivity encoding with arbitrary k-space trajectories.
          Magn Reson Med (2001).
    """

    def __init__(
        self,
        mask: torch.Tensor | np.ndarray,
        coil: torch.Tensor | np.ndarray,
        dcf: bool | None = False,
        verbose: bool | None = False,
        normalization: bool | None = False,
        device: torch.device | None = torch.device('cuda'),
    ) -> None:
        """Initialize the SENSE operator.

        Args:
            mask: Radial trajectory with shape [N_frames, N_spokes, N_samples,
                2]
            coil: Coil sensitivity maps with shape [1, N_coil, N_y, N_x]. If
                given, these will be multiplied before the forward NUFFT.
            dcf: Density compensation function flag. Defaults to False.
            verbose: Verbose flag for printing debug information.
                Defaults to False.
            normalization: Normalization flag for scaling the output.
                Defaults to False.
            device: Device to run the operations on. Defaults to 'cpu'.

        Raises:
            AttributeError: If device is not set to the cuda torch.device.
            OSError: If CUDA is not available or loaded.
        """
        # Set device to CUDA if available
        if device.type == 'cuda':
            sigpy.Device(0)

        # Convert arguments to tensors on device if they are numpy arrays
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.to(device)
        if isinstance(coil, np.ndarray):
            coil = torch.from_numpy(coil)
        coil = coil.to(device)

        # Check the shape of the mask
        assert (
            mask.dim() == 4 and mask.shape[-1] == 2
        ), 'Trajectory mask should have shape [N_frames, N_spokes, N_samples, 2] but got {}'.format(
            mask.shape
        )
        # Check the shape of the coil sensitivity maps
        assert (
            coil.dim() == 3
        ), 'Coil sensitivity maps should have shape [N_coil, N_y, N_x] but got {}'.format(
            coil.shape
        )

        self.mask = mask
        self.coil = coil
        self.dcf = dcf
        self.verbose = verbose
        self.normalization = normalization
        self.device = device

        self.N_frames, self.N_spokes, self.N_samples, _ = self.mask.shape
        self.ishape = [self.N_frames] + list(self.coil.shape[1:])

    def _get_normalization_scale(
        self, nrm_0: torch.Tensor, nrm_1: torch.Tensor, output_dim: int
    ) -> torch.Tensor:
        """Get normalization scale for the output."""
        if self.normalization:
            if torch.all(nrm_1 == 0):
                nrm_1 = nrm_1 + 1e-6
            scale = nrm_0 / nrm_1
            for _ in range(output_dim - 1):
                scale = scale.unsqueeze(1)
        else:
            scale = 1

        return scale

    def fwd(self, input: torch.Tensor | np.ndarray) -> torch.Tensor:
        """SENSE Forward Operator: from image to k-space.

        Args:
            input: Complex input image data with shape [N_frames, N_coil, N_y,
                N_x].

        Returns:
            Output k-space data with shape [N_frames, N_coil, N_spokes,
            N_samples, 2].
        """
        # Convert input to tensor, move to device, and ensure it's complex
        input = torch.as_tensor(
            input, device=self.device, dtype=torch.complex64
        )

        # Check the shape of the mask
        assert (
            input.dim() == 4
        ), 'Input should have shape [N_frames, N_coil, N_y, N_x] but got {}'.format(
            input.shape
        )

        output = []
        for t in range(self.N_frames):
            traj_t = self.mask[t]
            NUFFT_FWD = sigpy.linop.NUFFT(ishape=input.shape[1:], coord=traj_t)
            NUFFT_FWD = sigpy.to_pytorch_function(
                NUFFT_FWD, input_iscomplex=True, output_iscomplex=True
            )

            imag_t = input[t]
            imag_t = torch.view_as_real(imag_t)
            grid_t = NUFFT_FWD.apply(imag_t)
            grid_t = torch.view_as_complex(grid_t)

            output.append(grid_t)

        output = torch.stack(output).to(self.device)

        # Normalize the output if flag was set
        nrm_0 = torch.linalg.norm(input, dim=(-2, -1)).flatten()
        nrm_1 = torch.linalg.norm(output, dim=(-2, -1)).flatten()
        scale = self._get_normalization_scale(nrm_0, nrm_1, output.dim())
        output = output * scale

        return output

    def adj(
        self, input: torch.Tensor | np.ndarray, oshape: tuple | None = None
    ) -> torch.Tensor:
        """SENSE Adjoint Operator: from k-space to image.

        Args:
            input: Input k-space data with shape [N_frames, N_coils, N_spokes,
                N_samples].
            oshape: Output shape of the image. Defaults to None.

        Returns:
            Output image data with shape [N_frames, N_coil, N_y, N_x].
        """
        # Convert input to tensor, move to device, and ensure it's complex
        input = torch.as_tensor(
            input, device=self.device, dtype=torch.complex64
        )

        # Check the shape of the input data
        assert (
            input.dim() == 4
        ), 'Input data should have shape [N_frames, N_coils, N_spokes, N_samples] but got {}'.format(
            input.shape
        )

        if oshape is None:
            oshape = self.coil.shape

        output = []
        for t in range(self.N_frames):
            traj_t = self.mask[t]

            # density compensation function
            if self.dcf:
                # compute distance from 0, 0 coord for all traj coords
                # r = sqrt(y ** 2 + x ** 2)
                dcomp = torch.sqrt(torch.sum(traj_t**2, axis=-1))

                # normalize between 1/N_samples and 1
                margin = dcomp.max() - dcomp.min()
                scaling = (1 - 1 / self.N_samples) / margin
                dcomp = (dcomp - dcomp.min()) * scaling + 1 / self.N_samples

                # add pseudo coil dim
                dcomp = dcomp[None, :]
            else:
                dcomp = 1.0

            NUFFT_ADJ = sigpy.linop.NUFFTAdjoint(oshape=oshape, coord=traj_t)
            NUFFT_ADJ = sigpy.to_pytorch_function(
                NUFFT_ADJ, input_iscomplex=True, output_iscomplex=True
            )

            grid_t = input[t]
            grid_t = torch.view_as_real(grid_t * dcomp)
            imag_t = NUFFT_ADJ.apply(grid_t)
            imag_t = torch.view_as_complex(imag_t)

            output.append(imag_t)

        output = torch.stack(output).to(self.coil)

        # Normalize the output if flag was set
        nrm_0 = torch.linalg.norm(input, dim=(-2, -1)).flatten()
        nrm_1 = torch.linalg.norm(output, dim=(-2, -1)).flatten()
        scale = self._get_normalization_scale(nrm_0, nrm_1, output.dim())
        output = output * scale

        return output


class SenseSp:
    """Implementation of the SENSE Operator based on SigPy."""
