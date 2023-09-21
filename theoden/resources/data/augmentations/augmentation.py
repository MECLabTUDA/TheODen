import torch
import numpy as np

from abc import ABC, abstractmethod

from .. import Sample
from ....common import Transferable


class Augmentation(ABC, Transferable, is_base_type=True):
    """Abstract class for augmentations

    This class is used to augment samples. It is an abstract class and must be inherited from.
    It contains a function to augment a sample and a function to modify the metadata after augmenting.
    The modification of the metadata is optional. It should be used to update the relevant metadata fields after augmenting.
    """

    @staticmethod
    def _transform_to_numpy(tensor_image: torch.Tensor) -> np.ndarray:
        """Transforms a tensor image to a numpy array

        Args:
            tensor_image (torch.Tensor): Tensor image to transform

        Returns:
            np.ndarray: Numpy array of the image
        """
        return tensor_image.permute(1, 2, 0).numpy() * 255.0

    @staticmethod
    def _transform_to_tensor(numpy_image: np.ndarray) -> torch.Tensor:
        """Transforms a numpy array to a tensor image

        Args:
            numpy_image (np.ndarray): Numpy array to transform

        Returns:
            torch.Tensor: Tensor image of the numpy array
        """
        return torch.from_numpy(numpy_image).permute(2, 0, 1) / 255.0

    @abstractmethod
    def _augment(self, sample: Sample) -> Sample:
        """Function to augment sample

        Args:
            sample (Sample): Sample to augment

        Returns:
            Sample: Augmented sample
        """
        raise NotImplementedError("Please Implement this method")

    def _set_metadata(self, sample: Sample) -> Sample:
        """Function to modify metadata after augmenting

        Args:
            sample (Sample): Sample to modify

        Returns:
            Sample: Modified sample
        """
        return sample

    def __call__(self, sample: Sample) -> Sample:
        """Function call to augment sample

        Args:
            sample (Sample): Sample to augment

        Returns:
            Sample: Augmented sample with modified metadata
        """

        # sample = init_augmentation(sample)

        sample = self._augment(sample)

        sample = self._set_metadata(sample)

        return sample
