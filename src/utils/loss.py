"""
Copyright (c) Jinho Kim (jinho.kim@fau.de).
Copyright (c) Erik Goesche (erik.goesche@fau.de).

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn


class MixL1L2Loss(nn.Module):
    """Custom loss function that combines L1 and L2 loss."""

    def __init__(self, eps: float = 1e-10):
        """Initializes the loss function with optional parameters.

        Args:
            eps: A small value to avoid division by zero. Defaults to 1e-10.
        """
        super().__init__()
        self.eps = eps
        self.scaler = 0.5

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Computes a custom loss function which is a combination of the L2 and L1 norms.

        The loss is calculated as the average of the ratio of the L2 norm of the difference
        between the prediction and target to the L2 norm of the target, and the ratio of the
        L1 norm of the difference between the prediction and target to the L1 norm of the target.

        Args:
            pred: The predicted tensor.
            targ: The target tensor.

        Returns:
            The computed loss value.
        """
        return 0 * (
            torch.linalg.vector_norm(pred - targ)
            / torch.linalg.vector_norm(targ)
        ) + 1.0 * (
            torch.linalg.vector_norm(pred - targ, ord=1)
            / torch.linalg.vector_norm(targ, ord=1)
        )
