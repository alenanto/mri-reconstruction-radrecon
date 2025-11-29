import torch


def mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """Initializes a mask with the center filled in the second to last dimension.

    Args:
        x: The image / k-space to mask.
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[..., mask_from:mask_to, :] = x[..., mask_from:mask_to, :]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """Initializes a mask with the center filled in the second to last dimension.

    Can operate with different masks for each batch element.

    Args:
        x: The images / k-spaces to mask.
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError('mask_from and mask_to must match shapes.')
    if not mask_from.ndim == 1:
        raise ValueError('mask_from and mask_to must have 1 dimension.')
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError(
                'mask_from and mask_to must have batch_size length.'
            )

    if mask_from.shape[0] == 1:
        mask = mask_center(x, mask_from, mask_to)
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, ..., start:end, :] = x[i, ..., start:end, :]

    return mask
