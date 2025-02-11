# pylint: disable=abstract-method
import logging
from typing import Optional
import torch
from lightkit.data import DataLoader
from lightkit.utils import PathType
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import TensorDataset
from ._base import DataModule, OutputType
from ._registry import register

logger = logging.getLogger(__name__)

@register("FP")
class FPDataModule(DataModule):
    """
    Data module for the Sensorless Drive dataset.
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, root: Optional[PathType] = None, seed: Optional[int] = None):
        """
        Args:
            X: The input tensor of shape (N, d).
            Y: The categorical labels tensor of length N.
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.X = X
        self.Y = Y
        self.did_setup = False
        self._input_size = X.shape[1]
        self._num_classes = len(torch.unique(Y))

    @property
    def output_type(self) -> OutputType:
        return "categorical"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([self._input_size])

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            # Initialize datasets
            self.train_dataset = TensorDataset(self.X, self.Y)
            self.val_dataset = self.train_dataset

            # Mark done
            self.did_setup = True

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=1024, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=4096)
