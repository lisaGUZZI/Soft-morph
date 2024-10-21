from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMorphOperator2D(nn.Module, ABC):
    """
    Base class for soft morphological operations (erosion, dilation) using PyTorch.
    Contains shared logic such as input validation, unfolding the image, and iteration handling.
    Subclasses should implement the specific morphological operation in the 'apply_operation' method.
    """

    def __init__(self, indices_list, max_iter=1, connectivity=4):
        super(SoftMorphOperator2D, self).__init__()
        self._indices_list = indices_list
        self._max_iter = max_iter
        self._connectivity = connectivity

    def test_format(self, img):
        """
        Function to check user inputs:
        - Input image shape must either be [batch_size, channels, height, width] or [height, width].
        - Input image values must be between 0 and 1.
        - Connectivity represents the structuring element of the operation. In 2D, it must be either 4 or 8.
        """
        dim = img.dim()
        size = img.size()
        if dim > 4 or dim < 2:
            raise Exception(
                f"Invalid input shape {size}. Expected [batch_size, channels, height, width] or [height, width]."
            )
        elif dim < 4:
            if dim == 3:
                if size[0] > 3:
                    raise Exception(
                        f"Ambiguous input shape {size}. Expected [batch_size, channels, height, width] or [height, width]."
                    )
            for i in range(4 - dim):
                img = img.unsqueeze(0)
            print("Image resized to: ", img.size())
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Input image values must be in the range [0, 1].")
        if self._connectivity not in (4, 8):
            raise ValueError("Connectivity should either be 4 or 8.")
        return img

    @abstractmethod
    def forward(self, img):
        """
        Inputs:
        - im: Input 2D image of shape [batch_size, channels, height, width] or [height, width].
        Output: Image after morphological operation.
        """
        raise NotImplementedError(
            "Subclasses should implement the specific morphological operation."
        )
