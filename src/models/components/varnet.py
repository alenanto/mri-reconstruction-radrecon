"""
Variational network model.

Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Erik Goesche <erik.goesche@fau.de>.

Part of this code is licensed under the MIT License.
"""

import torch
from torch import nn

from src.models.components import mri

from src.models.components.unet import NormUnet, Unet2d, Unet2dPlusTime
from utils.math import c2r, r2c


def sens_expand(
    x: torch.Tensor, sens_maps: torch.Tensor, sense_op: mri.SenseOp
) -> torch.Tensor:
    """Expand single coil image data to multi-coil k-space data.
    Args:
        x: Complex-valued single coil image data of shape [1, N_frames, N_y,
            N_x].
        sens_maps: Sensitivity maps of shape [1, N_coils, N_y, N_x].
        op: SENSE operator.

    Returns:
        Multi-coil k-space data of shape [N_frames, N_coils, N_spokes,
        N_samples].
    """
    x_cc = x.permute(1, 0, 2, 3) * sens_maps
    return sense_op.fwd(x_cc)


def sens_reduce(
    x: torch.Tensor, sens_maps: torch.Tensor, sense_op: mri.SenseOp
) -> torch.Tensor:
    """Reduce multi-coil k-space data to single coil image data.
    Args:
        x: Multi-coil k-space data of shape [N_frames, N_coils, N_spokes,
            N_samples].
        sens_maps: Sensitivity maps of shape [1, N_coils, N_y, N_x].
        sense_op: SENSE operator.

    Returns:
        Single coil image data of shape [N_frames, N_y, N_x].
    """
    x_img = sense_op.adj(x)
    return torch.sum(x_img * sens_maps.conj(), dim=1)


class VarNet(nn.Module):
    """A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int,
        chans: int,
        pools: int,
        conv_size: int,
        conv_mode: str = '2d+t',
        use_sens_net: bool = False,
        sens_chans: int = 8,
        sens_pools: int = 4,
        mask_center: bool = True,
    ):
        """Initializes the `VarNet` model.
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            conv_size: Convolution kernel size in spatial dimension.
            conv_mode: Convolution mode ('2d', '2d+t').
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        # Instance sensitivity map U-Net if required
        self.sens_net = None

        # Define U-Net arguments for 2D and 2D+t modes
        unet_args = {
            'in_chans': 2,
            'out_chans': 2,
            'chans': chans,
            'num_pool_layers': pools,
            'conv_size': conv_size,
        }
        unet_spatiotemporal_args = {
            **unet_args,
            'pool_temporal': True,
            'conv_size_temp': 3,
            'pad_mode_temp': 'replicate',
        }

        # Mapping conv_mode to the appropriate class and args
        if conv_mode.lower() == '2d':
            unet_class = Unet2d
            unet_kwargs = unet_args
        elif conv_mode.lower() == '2d+t':
            unet_class = Unet2dPlusTime
            unet_kwargs = unet_spatiotemporal_args
        else:
            raise ValueError(f'Unknown conv_mode {conv_mode}')

        self.cascades = nn.ModuleList(
            [
                VarNetBlock(NormUnet(unet_class(**unet_kwargs)))
                for _ in range(num_cascades)
            ]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sens_maps: torch.Tensor | None,
        mask: torch.Tensor,
        num_low_frequencies: int | None = None,
    ) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            masked_kspace: Input k-space data of shape [N_batch, N_channel,
                N_frames, N_coils, N_spokes, N_samples].
            sens_maps: Sensitivity maps of shape [N_batch, N_coils, N_y, N_x].
            mask: Mask for k-space data of shape [N_batch, N_frames, N_spokes,
                N_samples, N_channel].
            num_low_frequencies: Number of low frequencies to use for coil
                sensitivity map calculation. Defaults to None.

        Returns:
            Reconstructed image data of shape [N_batch, N_channel, N_frames,
                N_y, N_x].
        """
        if self.sens_net is not None:
            sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        if sens_maps is None or sens_maps.numel() == 0:
            raise ValueError(
                'Sensitivity maps must be given if `use_sens_net` is False'
            )

        A = mri.SenseOp(
            mask.squeeze(0),  # Remove batch dimension
            sens_maps.squeeze(0),  # Remove batch dimension
            dcf=True,
            device=sens_maps.device,
        )
        # Create initial image prediction of shape [N_batch, N_frames, N_y, N_x]
        image_pred = sens_reduce(r2c(masked_kspace.squeeze(0)), sens_maps, A)
        image_pred = image_pred.unsqueeze(0)  # Add batch dimension

        for cascade in self.cascades:
            image_pred,dc = cascade(image_pred, masked_kspace, mask, sens_maps)

        return c2r(image_pred, axis=1),dc


class VarNetBlock(nn.Module):
    """Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """Initializes a `VarNetBlock`.
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_image: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model block.
        Args:
            current_image: Complex-valued current image of shape [N_batch,
            N_channel, N_frames, N_coils, N_spokes, N_samples].
            ref_kspace: Reference k-space data of shape [N_batch, N_channel,
                N_frames, N_coils, N_spokes, N_samples].
            mask: Mask for k-space data of shape [N_batch, N_frames, N_spokes,
                N_samples, N_channel].
            sens_maps: Coil sensitivity maps of shape [N_batch, N_coils, N_y,
                N_x].

        Returns:
            Updated image of shape [N_batch, N_channel, N_coils,
                N_spokes, N_samples].
        """
        A = mri.SenseOp(
            mask.squeeze(0),  # Remove batch dimension
            sens_maps.squeeze(0),  # Remove batch dimension
            dcf=True,
            device=sens_maps.device,
        )

        # Calculate soft data consistency term
        soft_dc = (
            sens_reduce(
                sens_expand(current_image, sens_maps, A)
                - r2c(ref_kspace.squeeze(0)),
                sens_maps,
                A,
            )
            * self.dc_weight
        )

        # Propagate current image through model to obtain refined image
        model_term = self.model(c2r(current_image.unsqueeze(0), axis=-1))

        return (
            current_image
            - soft_dc.unsqueeze(0)
            - r2c(model_term.squeeze(0), axis=-1)
        ),self.dc_weight
