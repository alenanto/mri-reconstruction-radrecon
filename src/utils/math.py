"""
Math utility functions for complex-valued images.

Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Zhengguo Tan <zhengguo.tan@gmail.com>.
Copyright (c) Erik Goesche <erik.goesche@fau.de>.

Part of this code is licensed under the MIT License.
"""

import os

import numpy as np
import numpy.typing as npt
import torch


def c2r(
    complex_img: npt.ArrayLike | torch.Tensor, axis: int = 0
) -> npt.ArrayLike | torch.Tensor:
    """Convert a complex-valued image to a real-valued image with real and
    imaginary parts along a specified axis.

    Args:
        complex_img (np.ndarray or torch.Tensor): Complex-valued image.
        axis: Axis along which to store the real and imaginary parts.

    Raises:
        NotImplementedError: If the input type is unsupported.

    Returns:
        Real-valued image.
    """
    if axis < -1 or axis >= complex_img.ndim:
        raise NotImplementedError('Axis out of range')
    if isinstance(complex_img, np.ndarray):
        # Extract real and imaginary parts
        real_part = complex_img.real
        imag_part = complex_img.imag
        # Stack along the specified axis
        real_img = np.stack((real_part, imag_part), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        # Extract real and imaginary parts
        real_part = complex_img.real
        imag_part = complex_img.imag
        # Stack along the specified axis
        real_img = torch.stack((real_part, imag_part), axis=axis)
    else:
        raise NotImplementedError('Unsupported type for complex_img')

    return real_img


def r2c(
    real_img: npt.ArrayLike | torch.Tensor, axis: int = 0
) -> npt.ArrayLike | torch.Tensor:
    """Convert a real-valued image with real and imaginary parts along a
    specified axis to a complex-valued image.

    Args:
        real_img: Real-valued image with shape (N, ..., 2, ...)
        axis: Axis along which the real and imaginary parts are stored.

    Raises:
        NotImplementedError: If the axis is out of range for the input array.

    Returns:
        np.ndarray: Complex-valued image.
    """
    if axis < -1 or axis >= real_img.ndim:
        raise NotImplementedError('Axis out of range')

    # Move the specified axis to the first dimension
    if isinstance(real_img, np.ndarray):
        real_img = np.moveaxis(real_img, axis, 0)
    elif isinstance(real_img, torch.Tensor):
        real_img = torch.moveaxis(real_img, axis, 0)
    else:
        raise NotImplementedError('Unsupported type for real_img')

    # Extract real and imaginary parts
    real_part = real_img[0]
    imag_part = real_img[1]

    # Create complex image
    complex_img = real_part + 1j * imag_part

    return complex_img


def ifftnc(
    data: torch.Tensor, norm: str = 'ortho', ndim: int = 2
) -> torch.Tensor:
    """Apply centered n-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data where dimension -1 has size two
            and represents the complex dimension.
        norm: Normalization mode. See ``torch.fft.ifft``.
        ndim: Number of back dimensions along which to apply the ifft
            (i.e. `ndim`-dimensional inverse Fourier transform),
            ignoring the last dimension of size 2.

    Returns:
        The IFFT of the input.
    """
    assert data.shape[-1] == 2, 'Tensor does not have separate complex dim.'
    assert ndim in (1, 2, 3), f'`ndim` must be in (1, 2, 3), but was {ndim}'

    if os.environ.get('ONNX_EXPORT', False) == 'True':
        raise NotImplementedError(
            'Centered n-dim FFT for ONNX export is currently not supported.'
        )
        # data = ortnum.fft.ifftshift_c2c(data, ndim=ndim)
        # data = ortnum.fft.ifft_c2c(data, ndim=ndim, norm=norm)
        # data = ortnum.fft.fftshift_c2c(data, ndim=ndim)
    else:
        dim = list(range(-ndim, 0))
        data = torch.view_as_complex(data)
        data = torch.fft.ifftshift(data, dim=dim)
        data = torch.fft.ifftn(data, dim=dim, norm=norm)
        data = torch.fft.fftshift(data, dim=dim)
        data = torch.view_as_real(data)

    return data


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError('Tensor does not have separate complex dim.')

    return (data**2).sum(dim=-1)


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data**2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))
