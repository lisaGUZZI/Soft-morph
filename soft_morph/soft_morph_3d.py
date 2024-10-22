import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from .base import SoftMorphOperator3D

class SoftErosion3D(SoftMorphOperator3D):

    def __init__(self, max_iter=1, connectivity=6):
        indices_list = [self.extract_indices(o) for o in range(1)]
        super().__init__(indices_list=indices_list, max_iter=max_iter, connectivity=connectivity)


    @staticmethod
    def extract_indices(o):
        """
        Function to extract extract ordered index list in each subdirection (North, East, South, West, Up, Down)
        """
        ind = [
            # Up
            torch.tensor(
                [
                    [2, 0, 0],
                    [2, 0, 1],
                    [2, 0, 2],
                    [1, 0, 2],
                    [0, 0, 2],
                    [0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [2, 1, 0],
                    [2, 1, 1],
                    [2, 1, 2],
                    [1, 1, 2],
                    [0, 1, 2],
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 1, 0],
                    [2, 2, 0],
                    [2, 2, 1],
                    [2, 2, 2],
                    [1, 2, 2],
                    [0, 2, 2],
                    [0, 2, 1],
                    [0, 2, 0],
                    [1, 2, 0],
                    [1, 2, 1],
                    [1, 1, 1]
                ],
                dtype=torch.long,
            )
        ]

        indices = ind[o]

        return indices
    
    
    def apply_transformation(self, n):
        """
        Apply polynomial formula based on the boolean expression that defines an erosion on each 3x3x3 overlapping cubes of the 3D image.
        Inputs : vector of 3x3x3 overlapping cubes n, connectivity (6,18 or 26) defining the structuring element.
        Output : In binary case returns 0 if the central pixel needs to be changed to 0, returns 1 otherwise.
        """
        if self._connectivity == 6:
            vox = [8, 10, 12, 25, 16, 14, 26]
        elif self._connectivity == 18:
            vox = [8, 10, 12, 25, 16, 14, 1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 22, 24, 26]
        else:
            vox = [
                8,
                10,
                12,
                25,
                16,
                14,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                18,
                20,
                22,
                24,
                0,
                2,
                4,
                6,
                17,
                19,
                21,
                23,
                26,
            ]

        F = torch.prod(n[:, :, :, vox], dim=-1)

        return F

    def forward(self, img):
        """
        Inputs :
        - im : input 3D image of shape [batch_size, channels, depth, height, width] or [depth, height, width].
        - iterations : number of times the morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 6, 18 or 26.
        Output : Image after morphological operation
        """
        img = self.test_format(img)  # Check user inputs
        for _ in range(self._max_iter):
            unfolded = torch.nn.functional.pad(
                img, (1, 1, 1, 1, 1, 1), mode="constant", value=1
            )
            unfolded = (
                unfolded.unfold(2,3, 1)
                .unfold(3,3, 1)
                .unfold(4,3, 1)
            )
            unfolded = unfolded.contiguous().view(
                img.shape[0],
                img.shape[1],
                (img.shape[2] * img.shape[3] * img.shape[4]),
                (3**3),
            )
            # Apply the formula to all windows simultaneously
            unfolded = unfolded[
                :,
                :,
                :,
                (self._indices_list[0][:, 0] * 9)
                + (self._indices_list[0][:, 1] * 3)
                + self._indices_list[0][:, 2],
            ]
            output = self.apply_transformation(unfolded)
            # Adjust the dimensions of output to match the spatial dimensions of img
            output = output.view(
                output.size(0), output.size(1), img.shape[2], img.shape[3], img.shape[4]
            )
            # Element-wise multiplication
            img = img * output
        return img


class SoftDilation3D(SoftMorphOperator3D):

    def __init__(self, max_iter=1, connectivity=6):
        indices_list = [self.extract_indices(o) for o in range(1)]
        super().__init__(indices_list=indices_list, max_iter=max_iter, connectivity=connectivity)

    @staticmethod
    def extract_indices(o):
        """
        Function to extract extract ordered index list in each subdirection (North, East, South, West, Up, Down)
        """
        ind = [
            # Up
            torch.tensor(
                [
                    [2, 0, 0],
                    [2, 0, 1],
                    [2, 0, 2],
                    [1, 0, 2],
                    [0, 0, 2],
                    [0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [2, 1, 0],
                    [2, 1, 1],
                    [2, 1, 2],
                    [1, 1, 2],
                    [0, 1, 2],
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 1, 0],
                    [2, 2, 0],
                    [2, 2, 1],
                    [2, 2, 2],
                    [1, 2, 2],
                    [0, 2, 2],
                    [0, 2, 1],
                    [0, 2, 0],
                    [1, 2, 0],
                    [1, 2, 1],
                    [1, 1, 1]
                ],
                dtype=torch.long,
            )
        ]

        indices = ind[o]

        return indices
    
    def apply_transformation(self, n):
        """
        Apply polynomial formula based on the boolean expression that defines a dilation on each 3x3x3 overlapping cubes of the 3D image.
        Inputs : vector of 3x3x3 overlapping cubes n, connectivity (6, 18 or 26) defining the structuring element.
        Output : Returns the new value attributed to each central pixel
        """
        if self._connectivity == 6:
            vox = [8, 10, 12, 25, 16, 14, 26]
        elif self._connectivity == 18:
            vox = [8, 10, 12, 25, 16, 14, 1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 22, 24, 26]
        else:
            vox = [
                8,
                10,
                12,
                25,
                16,
                14,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                18,
                20,
                22,
                24,
                0,
                2,
                4,
                6,
                17,
                19,
                21,
                23,
                26,
            ]

        F = torch.prod(1 - n[:, :, :, vox], dim=-1)
        return 1 - F

    def forward(self, img):
        """
        Inputs :
        - im : input 3D image of shape [batch_size, channels, depth, height, width] or [depth, height, width].
        - iterations : number of times the morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 6, 18 or 26.
        Output : Image after morphological operation
        """
        img = self.test_format(img)
        for _ in range(self._max_iter):
            unfolded = torch.nn.functional.pad(
                img, (1, 1, 1, 1, 1, 1), mode="constant", value=0
            )
            unfolded = (
                unfolded.unfold(2,3, 1)
                .unfold(3,3, 1)
                .unfold(4,3, 1)
            )
            unfolded = unfolded.contiguous().view(
                img.shape[0],
                img.shape[1],
                (img.shape[2] * img.shape[3] * img.shape[4]),
                (3**3),
            )
            # Apply the formula to all windows simultaneously
            unfolded = unfolded[
                :,
                :,
                :,
                (self._indices_list[0][:, 0] * 9)
                + (self._indices_list[0][:, 1] * 3)
                + self._indices_list[0][:, 2],
            ]
            output = self.apply_transformation(unfolded)
            # Adjust the dimensions of output to match the spatial dimensions of img
            img = output.view(
                output.size(0), output.size(1), img.shape[2], img.shape[3], img.shape[4]
            )
        return img


class SoftMorphTransform3D(nn.Module, ABC):
    """
    Base class for soft morphological operations (opening, closing) using PyTorch.
    Provides a framework to apply erosion followed by dilation (or vice versa)
    with customizable connectivity and iterations.
    """

    def __init__(self, max_iter=1, dilation_connectivity=6, erosion_connectivity=6):
        super(SoftMorphTransform3D, self).__init__()
        self.dilate = SoftDilation3D(max_iter, dilation_connectivity)
        self.erode = SoftErosion3D(max_iter, erosion_connectivity)

    @abstractmethod
    def forward(self, img):
        raise NotImplementedError(
            "forward method must be implemented in derived classes"
        )

class SoftClosing3D(SoftMorphTransform3D):
    """
    Class implemented using Pytorch module to perform differentiable soft closing on 3D input image.
    """
    
    def forward(self, img):
        """
        Inputs :
        - img : input 3D image of shape [batch_size, channels, depth, height, width] or [depth, height, width].
        Output : Image after morphological operation
        """
        output = self.dilate(img)
        output = self.erode(output)
        return output


class SoftOpening3D(SoftMorphTransform3D):
    """
    Class implemented using Pytorch module to perform differentiable soft opening on 3D input image.
    """
    
    def forward(self, img):
        """
        Inputs :
        - img : input 3D image of shape [batch_size, channels, depth, height, width] or [depth, height, width].
        Output : Image after morphological operation
        """
        output = self.erode(img)
        output = self.dilate(output)
        return output


class SoftSkeletonizer3D(SoftMorphOperator3D):
    """
    Class implemented using Pytorch module to perform differentiable soft skeletonization on 3D input image.

    the max_iter input represents the number of times the thinning operation will be repeated.
    This input will be automatically determined in future versions.
    """

    def __init__(self, max_iter=5):
        # Extract ordered index list in each subdirection (Up, East, South, Down, West, North)
        indices_list = [self.extract_indices(o) for o in range(6)]
        super().__init__(indices_list=indices_list, max_iter=max_iter)

    @staticmethod
    def extract_indices(o):
        """
        Function to extract extract ordered index list in each subdirection (North, East, South, West, Up, Down)
        """
        ind = [
            # Up
            torch.tensor(
                [
                    [2, 0, 0],
                    [2, 0, 1],
                    [2, 0, 2],
                    [1, 0, 2],
                    [0, 0, 2],
                    [0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [2, 1, 0],
                    [2, 1, 1],
                    [2, 1, 2],
                    [1, 1, 2],
                    [0, 1, 2],
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 1, 0],
                    [2, 2, 0],
                    [2, 2, 1],
                    [2, 2, 2],
                    [1, 2, 2],
                    [0, 2, 2],
                    [0, 2, 1],
                    [0, 2, 0],
                    [1, 2, 0],
                    [1, 2, 1],
                ],
                dtype=torch.long,
            ),
            # East
            torch.tensor(
                [
                    [2, 0, 2],
                    [2, 1, 2],
                    [2, 2, 2],
                    [1, 2, 2],
                    [0, 2, 2],
                    [0, 1, 2],
                    [0, 0, 2],
                    [1, 0, 2],
                    [1, 1, 2],
                    [2, 0, 1],
                    [2, 1, 1],
                    [2, 2, 1],
                    [1, 2, 1],
                    [0, 2, 1],
                    [0, 1, 1],
                    [0, 0, 1],
                    [1, 0, 1],
                    [2, 0, 0],
                    [2, 1, 0],
                    [2, 2, 0],
                    [1, 2, 0],
                    [0, 2, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                ],
                dtype=torch.long,
            ),
            # South
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 1, 2],
                    [0, 2, 2],
                    [0, 2, 1],
                    [0, 2, 0],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 0, 2],
                    [1, 1, 2],
                    [1, 2, 2],
                    [1, 2, 1],
                    [1, 2, 0],
                    [1, 1, 0],
                    [2, 0, 0],
                    [2, 0, 1],
                    [2, 0, 2],
                    [2, 1, 2],
                    [2, 2, 2],
                    [2, 2, 1],
                    [2, 2, 0],
                    [2, 1, 0],
                    [2, 1, 1],
                ],
                dtype=torch.long,
            ),
            # down
            torch.tensor(
                [
                    [0, 2, 0],
                    [0, 2, 1],
                    [0, 2, 2],
                    [1, 2, 2],
                    [2, 2, 2],
                    [2, 2, 1],
                    [2, 2, 0],
                    [1, 2, 0],
                    [1, 2, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 2],
                    [1, 1, 2],
                    [2, 1, 2],
                    [2, 1, 1],
                    [2, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [1, 0, 2],
                    [2, 0, 2],
                    [2, 0, 1],
                    [2, 0, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                ],
                dtype=torch.long,
            ),
            # West
            torch.tensor(
                [
                    [2, 2, 0],
                    [2, 1, 0],
                    [2, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 2, 0],
                    [1, 2, 0],
                    [1, 1, 0],
                    [2, 2, 1],
                    [2, 1, 1],
                    [2, 0, 1],
                    [1, 0, 1],
                    [0, 0, 1],
                    [0, 1, 1],
                    [0, 2, 1],
                    [1, 2, 1],
                    [2, 2, 2],
                    [2, 1, 2],
                    [2, 0, 2],
                    [1, 0, 2],
                    [0, 0, 2],
                    [0, 1, 2],
                    [0, 2, 2],
                    [1, 2, 2],
                    [1, 1, 2],
                ],
                dtype=torch.long,
            ),
            # North
            torch.tensor(
                [
                    [2, 2, 0],
                    [2, 2, 1],
                    [2, 2, 2],
                    [2, 1, 2],
                    [2, 0, 2],
                    [2, 0, 1],
                    [2, 0, 0],
                    [2, 1, 0],
                    [2, 1, 1],
                    [1, 2, 0],
                    [1, 2, 1],
                    [1, 2, 2],
                    [1, 1, 2],
                    [1, 0, 2],
                    [1, 0, 1],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 2, 0],
                    [0, 2, 1],
                    [0, 2, 2],
                    [0, 1, 2],
                    [0, 0, 2],
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 1, 1],
                ],
                dtype=torch.long,
            ),
        ]

        indices = ind[o]

        return indices

    def apply_transformation(self, n):
        """
        Apply polynomial formula based on the boolean expression that defines a thinning operation on each 3x3x3 overlapping cubes of the 3D image.
        Inputs : vector of 3x3x3 overlapping cubes n.
        Output : In binary case returns 0 if the central pixel needs to be changed to 0, returns 1 otherwise.
        """
        M1 = (
            (1 - n[:, :, :, 0])
            * (1 - n[:, :, :, 1])
            * (1 - n[:, :, :, 2])
            * (1 - n[:, :, :, 3])
            * (1 - n[:, :, :, 4])
            * (1 - n[:, :, :, 5])
            * (1 - n[:, :, :, 6])
            * (1 - n[:, :, :, 7])
            * (1 - n[:, :, :, 8])
            * n[:, :, :, 25]
            * (
                1
                - (
                    (1 - n[:, :, :, 9])
                    * (1 - n[:, :, :, 10])
                    * (1 - n[:, :, :, 11])
                    * (1 - n[:, :, :, 12])
                    * (1 - n[:, :, :, 13])
                    * (1 - n[:, :, :, 14])
                    * (1 - n[:, :, :, 15])
                    * (1 - n[:, :, :, 16])
                    * (1 - n[:, :, :, 17])
                    * (1 - n[:, :, :, 18])
                    * (1 - n[:, :, :, 19])
                    * (1 - n[:, :, :, 20])
                    * (1 - n[:, :, :, 21])
                    * (1 - n[:, :, :, 22])
                    * (1 - n[:, :, :, 23])
                    * (1 - n[:, :, :, 24])
                )
            )
        )
        M2 = ((1 - n[:, :, :, 8]) * n[:, :, :, 25]) * (
            1
            - (
                (
                    1
                    - (
                        (1 - n[:, :, :, 3])
                        * (1 - n[:, :, :, 4])
                        * (1 - n[:, :, :, 5])
                        * (1 - n[:, :, :, 6])
                        * (1 - n[:, :, :, 7])
                        * n[:, :, :, 10]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 1])
                        * (1 - n[:, :, :, 2])
                        * (1 - n[:, :, :, 3])
                        * (1 - n[:, :, :, 4])
                        * (1 - n[:, :, :, 5])
                        * n[:, :, :, 16]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 0])
                        * (1 - n[:, :, :, 1])
                        * (1 - n[:, :, :, 2])
                        * (1 - n[:, :, :, 3])
                        * (1 - n[:, :, :, 7])
                        * n[:, :, :, 14]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 0])
                        * (1 - n[:, :, :, 1])
                        * (1 - n[:, :, :, 5])
                        * (1 - n[:, :, :, 6])
                        * (1 - n[:, :, :, 7])
                        * n[:, :, :, 12]
                    )
                )
            )
        )
        M3 = ((1 - n[:, :, :, 8]) * n[:, :, :, 25]) * (
            1
            - (
                (
                    1
                    - (
                        (1 - n[:, :, :, 5])
                        * (1 - n[:, :, :, 6])
                        * (1 - n[:, :, :, 7])
                        * n[:, :, :, 10]
                        * n[:, :, :, 12]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 5])
                        * (1 - n[:, :, :, 4])
                        * (1 - n[:, :, :, 3])
                        * n[:, :, :, 10]
                        * n[:, :, :, 16]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 1])
                        * (1 - n[:, :, :, 2])
                        * (1 - n[:, :, :, 3])
                        * n[:, :, :, 16]
                        * n[:, :, :, 14]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 0])
                        * (1 - n[:, :, :, 1])
                        * (1 - n[:, :, :, 7])
                        * n[:, :, :, 14]
                        * n[:, :, :, 12]
                    )
                )
            )
        )
        M4 = (
            (1 - n[:, :, :, 1])
            * (1 - n[:, :, :, 3])
            * (1 - n[:, :, :, 5])
            * (1 - n[:, :, :, 7])
            * (1 - n[:, :, :, 8])
            * n[:, :, :, 25]
        ) * (
            1
            - (
                (
                    1
                    - (
                        (1 - n[:, :, :, 0])
                        * (1 - n[:, :, :, 4])
                        * (1 - n[:, :, :, 6])
                        * n[:, :, :, 11]
                        * n[:, :, :, 2]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 2])
                        * (1 - n[:, :, :, 4])
                        * (1 - n[:, :, :, 6])
                        * n[:, :, :, 0]
                        * n[:, :, :, 9]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 0])
                        * (1 - n[:, :, :, 4])
                        * (1 - n[:, :, :, 2])
                        * n[:, :, :, 6]
                        * n[:, :, :, 15]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 0])
                        * (1 - n[:, :, :, 6])
                        * (1 - n[:, :, :, 2])
                        * n[:, :, :, 4]
                        * n[:, :, :, 13]
                    )
                )
            )
        )
        M5 = (
            (1 - n[:, :, :, 0])
            * (1 - n[:, :, :, 1])
            * (1 - n[:, :, :, 2])
            * (1 - n[:, :, :, 3])
            * (1 - n[:, :, :, 4])
            * (1 - n[:, :, :, 5])
            * (1 - n[:, :, :, 6])
            * (1 - n[:, :, :, 7])
            * (1 - n[:, :, :, 8])
            * (1 - n[:, :, :, 25])
        ) * (
            1
            - (
                (
                    1
                    - (
                        (1 - n[:, :, :, 13])
                        * (1 - n[:, :, :, 14])
                        * (1 - n[:, :, :, 15])
                        * (1 - n[:, :, :, 21])
                        * (1 - n[:, :, :, 22])
                        * (1 - n[:, :, :, 23])
                        * n[:, :, :, 18]
                        * (
                            1
                            - (
                                (1 - n[:, :, :, 9])
                                * (1 - n[:, :, :, 10])
                                * (1 - n[:, :, :, 11])
                                * (1 - n[:, :, :, 12])
                                * (1 - n[:, :, :, 16])
                                * (1 - n[:, :, :, 17])
                                * (1 - n[:, :, :, 19])
                                * (1 - n[:, :, :, 20])
                                * (1 - n[:, :, :, 24])
                            )
                        )
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 13])
                        * (1 - n[:, :, :, 12])
                        * (1 - n[:, :, :, 11])
                        * (1 - n[:, :, :, 21])
                        * (1 - n[:, :, :, 20])
                        * (1 - n[:, :, :, 19])
                        * n[:, :, :, 24]
                        * (
                            1
                            - (
                                (1 - n[:, :, :, 9])
                                * (1 - n[:, :, :, 10])
                                * (1 - n[:, :, :, 14])
                                * (1 - n[:, :, :, 15])
                                * (1 - n[:, :, :, 16])
                                * (1 - n[:, :, :, 17])
                                * (1 - n[:, :, :, 18])
                                * (1 - n[:, :, :, 22])
                                * (1 - n[:, :, :, 23])
                            )
                        )
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 9])
                        * (1 - n[:, :, :, 10])
                        * (1 - n[:, :, :, 11])
                        * (1 - n[:, :, :, 17])
                        * (1 - n[:, :, :, 18])
                        * (1 - n[:, :, :, 19])
                        * n[:, :, :, 22]
                        * (
                            1
                            - (
                                (1 - n[:, :, :, 15])
                                * (1 - n[:, :, :, 14])
                                * (1 - n[:, :, :, 13])
                                * (1 - n[:, :, :, 12])
                                * (1 - n[:, :, :, 16])
                                * (1 - n[:, :, :, 23])
                                * (1 - n[:, :, :, 21])
                                * (1 - n[:, :, :, 20])
                                * (1 - n[:, :, :, 24])
                            )
                        )
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 9])
                        * (1 - n[:, :, :, 16])
                        * (1 - n[:, :, :, 15])
                        * (1 - n[:, :, :, 24])
                        * (1 - n[:, :, :, 17])
                        * (1 - n[:, :, :, 23])
                        * n[:, :, :, 20]
                        * (
                            1
                            - (
                                (1 - n[:, :, :, 14])
                                * (1 - n[:, :, :, 10])
                                * (1 - n[:, :, :, 11])
                                * (1 - n[:, :, :, 12])
                                * (1 - n[:, :, :, 13])
                                * (1 - n[:, :, :, 18])
                                * (1 - n[:, :, :, 19])
                                * (1 - n[:, :, :, 22])
                                * (1 - n[:, :, :, 21])
                            )
                        )
                    )
                )
            )
        )
        M6 = (
            (1 - n[:, :, :, 0])
            * (1 - n[:, :, :, 1])
            * (1 - n[:, :, :, 2])
            * (1 - n[:, :, :, 3])
            * (1 - n[:, :, :, 4])
            * (1 - n[:, :, :, 5])
            * (1 - n[:, :, :, 6])
            * (1 - n[:, :, :, 7])
            * (1 - n[:, :, :, 8])
            * (1 - n[:, :, :, 25])
        ) * (
            1
            - (
                (
                    1
                    - (
                        (1 - n[:, :, :, 14])
                        * (1 - n[:, :, :, 15])
                        * (1 - n[:, :, :, 16])
                        * (1 - n[:, :, :, 22])
                        * (1 - n[:, :, :, 23])
                        * (1 - n[:, :, :, 24])
                        * n[:, :, :, 18]
                        * n[:, :, :, 20]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 14])
                        * (1 - n[:, :, :, 12])
                        * (1 - n[:, :, :, 13])
                        * (1 - n[:, :, :, 22])
                        * (1 - n[:, :, :, 20])
                        * (1 - n[:, :, :, 21])
                        * n[:, :, :, 18]
                        * n[:, :, :, 24]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 10])
                        * (1 - n[:, :, :, 11])
                        * (1 - n[:, :, :, 12])
                        * (1 - n[:, :, :, 18])
                        * (1 - n[:, :, :, 19])
                        * (1 - n[:, :, :, 20])
                        * n[:, :, :, 24]
                        * n[:, :, :, 22]
                    )
                )
                * (
                    1
                    - (
                        (1 - n[:, :, :, 9])
                        * (1 - n[:, :, :, 10])
                        * (1 - n[:, :, :, 16])
                        * (1 - n[:, :, :, 24])
                        * (1 - n[:, :, :, 17])
                        * (1 - n[:, :, :, 18])
                        * n[:, :, :, 22]
                        * n[:, :, :, 20]
                    )
                )
            )
        )

        F = 1 - ((1 - M1) * (1 - M2) * (1 - M3) * (1 - M4) * (1 - M5) * (1 - M6))
        F = 1 - F
        return F

    def forward(self, img):
        """
        Input :
        - img : input 3D image of shape [batch_size, channels, depth, height, width] or [depth, height, width].
        Output : Image after morphological operation
        """
        img = self.test_format(img)
        for _ in range(self._max_iter):
            for o in range(6):  # Iterate over all 6 orientations
                unfolded = torch.nn.functional.pad(
                    img, (1, 1, 1, 1, 1, 1), mode="constant", value=0
                )
                # Reshape to get every 3x3x3 overlapping squares with a stride of 1
                unfolded = (
                    unfolded.unfold(2,3, 1)
                    .unfold(3,3, 1)
                    .unfold(4,3, 1)
                )
                unfolded = unfolded.contiguous().view(
                    img.shape[0],
                    img.shape[1],
                    (img.shape[2] * img.shape[3] * img.shape[4]),
                    (3**3),
                )
                # Apply the formula to all cubes simultaneously
                unfolded = unfolded[
                    :,
                    :,
                    :,
                    (self._indices_list[o][:, 0] * 9)
                    + (self._indices_list[o][:, 1] * 3)
                    + self._indices_list[o][:, 2],
                ]
                output = self.apply_transformation(unfolded)
                # # Adjust the dimensions of output to match the spatial dimensions of img
                output = output.view(
                    output.size(0),
                    output.size(1),
                    img.shape[2],
                    img.shape[3],
                    img.shape[4],
                )
                # Element-wise multiplication
                img = img * output
        return img
