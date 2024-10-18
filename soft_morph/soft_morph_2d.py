import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SoftMorphOperator2D


class SoftErosion2D(SoftMorphOperator2D):

    def __init__(self, max_iter=1, connectivity=4):
        indices_list = torch.tensor(
            [[0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [1, 0], [0, 0], [1, 1]],
            dtype=torch.long,
        )
        super().__init__(indices_list=indices_list, max_iter=1, connectivity=4)

    def apply_transformation(self, n):
        """
        Apply polynomial formula based on the boolean expression that defines an erosion on each 3x3 overlapping squares of the 2D image.
        Inputs : vector of 3x3 overlapping squares n, connectivity (4 or 8) defining the structuring element.
        Output : In binary case returns 0 if the central pixel needs to be changed to 0, returns 1 otherwise.
        """
        if self._connectivity == 4:
            F = torch.prod(n[:, :, :, ::2], dim=-1)
        else:
            F = torch.prod(n, dim=-1)
        return F

    def forward(self, img):
        """
        Inputs :
        - im : input 2D image of shape [batch_size, channels, height, width] or [height, width].
        - iterations : number of times the morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 4 or 8.
        Output : Image after morphological operation
        """
        img = self.test_format(img)  # Check user inputs
        for _ in range(self._max_iter):
            img_padded = F.pad(img, (1, 1, 1, 1), mode="constant", value=1)
            # Unfold the tensor to extract overlapping 3x3 windows
            unf = nn.Unfold((img.shape[2], img.shape[3]), 1, 0, 1)
            unfolded = unf(img_padded)
            unfolded = unfolded.view(img.shape[0], img.shape[1], -1, unfolded.size(-1))
            # Apply the morphological operation formula to all windows simultaneously
            unfolded = unfolded[
                :, :, :, (self._indices_list[:, 0] * 3) + self._indices_list[:, 1]
            ]
            output = self.apply_transformation(unfolded)
            # Adjust the dimensions of output to match the spatial dimensions of im
            output = output.view(
                output.size(0), output.size(1), img.shape[2], img.shape[3]
            )
            # Element-wise multiplication
            img = img * output
        return img


class SoftDilation2D(SoftMorphOperator2D):

    def __init__(self, max_iter=1, connectivity=4):
        indices_list = torch.tensor(
            [[0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [1, 0], [0, 0], [1, 1]],
            dtype=torch.long,
        )

        super().__init__(indices_list, max_iter, connectivity)

    def apply_transformation(self, n):
        """
        Apply polynomial formula based on the boolean expression that defines a dilation on each 3x3 overlapping squares of the 2D image.
        Inputs : vector of 3x3 overlapping squares n, connectivity (4 or 8) defining the structuring element.
        Output : Returns the new value attributed to each central pixel
        """
        if self._connectivity == 4:
            F = 1 - torch.prod(1 - n[:, :, :, ::2], dim=-1)
        else:
            F = 1 - torch.prod(1 - n, dim=-1)
        return F

    def forward(self, img):
        """
        Inputs :
        - img : input 2D image of shape [batch_size, channels, height, width] or [height, width].
        - iterations : number of times the morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 4 or 8.
        Output : Image after morphological operation
        """
        img = self.test_format(img)
        for _ in range(self._max_iter):
            # Unfold the tensor to extract overlapping 3x3 windows
            unf = nn.Unfold((img.shape[2], img.shape[3]), 1, 1, 1)
            unfolded = unf(img)
            unfolded = unfolded.view(img.shape[0], img.shape[1], -1, unfolded.size(-1))
            # Apply the formula to all windows simultaneously
            unfolded = unfolded[
                :, :, :, (self._indices_list[:, 0] * 3) + self._indices_list[:, 1]
            ]
            output = self.apply_transformation(unfolded)
            # Adjust the dimensions of output to match the spatial dimensions of im
            img = output.view(
                output.size(0), output.size(1), img.shape[2], img.shape[3]
            )
        return img


class SoftMorphTransform2D(nn.Module):
    """
    Base class for soft morphological operations (opening, closing) using PyTorch.
    Provides a framework to apply erosion followed by dilation (or vice versa)
    with customizable connectivity and iterations.
    """

    def __init__(self, max_iter=1, dilation_connectivity=4, erosion_connectivity=4):
        super(SoftMorphTransform2D, self).__init__()
        self.dilate = SoftDilation2D(max_iter, dilation_connectivity)
        self.erode = SoftErosion2D(max_iter, erosion_connectivity)

    def apply_operations(self, input_img, order):
        """
        Applies the morphological operations in the given order.

        Inputs:
        - input_img: 2D input image of shape [batch_size, channels, height, width] or [height, width].
        - iterations: Number of iterations for the operations.
        - dilation_connectivity: Connectivity for dilation (4 or 8).
        - erosion_connectivity: Connectivity for erosion (4 or 8).
        - order: List defining the operation order, e.g., ['erode', 'dilate'] or ['dilate', 'erode'].

        Output:
        - Image after applying the morphological operations.
        """
        output = input_img
        for operation in order:
            if operation == "dilate":
                output = self.dilate(output)
            elif operation == "erode":
                output = self.erode(output)
        return output

    def forward(
        self,
        img,
        order=None,
    ):
        if order is None:
            raise NotImplementedError(
                "Subclasses should implement the operation order."
            )
        return self.apply_operations(img, order)


class SoftClosing(SoftMorphTransform2D):
    """
    SoftClosing operation: Dilation followed by Erosion.
    """

    def forward(
        self,
        input_img,
    ):
        order = ["dilate", "erode"]
        return super().forward(input_img, order)


class SoftOpening(SoftMorphTransform2D):
    """
    SoftOpening operation: Erosion followed by Dilation.
    """

    def forward(self, input_img):
        order = ["erode", "dilate"]
        return super().forward(input_img, order)


class SoftSkeletonizer(SoftMorphOperator2D):
    """
    Class implemented using Pytorch module to perform differentiable soft skeletonization on 2D input image.

    the max_iter input represents the number of times the thinning operation will be repeated.
    This input will be automatically determined in future versions.
    """

    def __init__(self, max_iter=5):
        indices_list = [self.extract_indices(o) for o in range(4)]
        super().__init__(indices_list=indices_list, max_iter=max_iter)

    @staticmethod
    def extract_indices(o):
        """
        Function to extract extract ordered index list in each subdirection (North, East, South, West)
        """
        indices = torch.tensor(
            [[0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [1, 0], [0, 0]],
            dtype=torch.long,
        )
        # Adjust indices based on orientation
        indices = torch.roll(indices, -2 * o, dims=0)

        return indices

    def apply_transformation(self, n):
        """
        Apply polynomial formula based on the boolean expression that defines a thinning operation on each 3x3 overlapping squares of the 2D image.
        Inputs : vector of 3x3 overlapping squares n.
        Output : In binary case returns 0 if the central pixel needs to be changed to 0, returns 1 otherwise.
        """
        F1 = 1 - n[:, :, :, 0]
        F2 = (
            (1 - n[:, :, :, 1])
            * (1 - n[:, :, :, 7])
            * (
                1
                - n[:, :, :, 2]
                - n[:, :, :, 3]
                + 2 * n[:, :, :, 2] * n[:, :, :, 3]
                - n[:, :, :, 4]
                + 2 * n[:, :, :, 2] * n[:, :, :, 4]
                + 2 * n[:, :, :, 3] * n[:, :, :, 4]
                - 4 * n[:, :, :, 2] * n[:, :, :, 3] * n[:, :, :, 4]
            )
            * (n[:, :, :, 3] + n[:, :, :, 5] - 2 * (n[:, :, :, 3] * n[:, :, :, 5]))
            * (n[:, :, :, 3] + n[:, :, :, 6] - 2 * (n[:, :, :, 3] * n[:, :, :, 6]))
        )
        F3 = (
            (n[:, :, :, 1] + n[:, :, :, 5] - 2 * (n[:, :, :, 1] * n[:, :, :, 5]))
            * (n[:, :, :, 2] + n[:, :, :, 5] - 2 * (n[:, :, :, 2] * n[:, :, :, 5]))
            * (
                n[:, :, :, 4]
                + (1 - n[:, :, :, 5])
                - 2 * (n[:, :, :, 4] * (1 - n[:, :, :, 5]))
            )
            * (1 - n[:, :, :, 6])
            * (1 - n[:, :, :, 7])
        )
        F4 = n[:, :, :, 2] * n[:, :, :, 4] * (1 - n[:, :, :, 7])
        F5 = (1 - n[:, :, :, 1]) * n[:, :, :, 4] * n[:, :, :, 6]
        F6 = (
            (1 - n[:, :, :, 1])
            * (1 - n[:, :, :, 2])
            * (1 - n[:, :, :, 3])
            * n[:, :, :, 6]
            * n[:, :, :, 7]
        )
        F = 1 - (F1 * (1 - ((1 - F2) * (1 - F3) * (1 - F4) * (1 - F5) * (1 - F6))))
        return F

    def forward(self, img):
        """
        Input :
        - im : input 2D image of shape [batch_size, channels, height, width] or [height, width].
        Output : Image after morphological operation
        """
        img = self.test_format(img)
        for _ in range(self._max_iter):
            for o in range(4):  # Iterate through all directions
                # Unfold the tensor to extract overlapping 3x3 windows
                unf = nn.Unfold((img.shape[2], img.shape[3]), 1, 1, 1)
                unfolded = unf(img)
                unfolded = unfolded.view(
                    img.shape[0], img.shape[1], -1, unfolded.size(-1)
                )
                # Apply the formula to all windows simultaneously
                unfolded = unfolded[
                    :,
                    :,
                    :,
                    (self._indices_list[o][:, 0] * 3) + self._indices_list[o][:, 1],
                ]
                output = self.apply_transformation(unfolded)
                # Adjust the dimensions of output to match the spatial dimensions of im
                output = output.view(
                    output.size(0), output.size(1), img.shape[2], img.shape[3]
                )
                # Element-wise multiplication
                img = img * output
        return img
