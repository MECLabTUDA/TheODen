import numpy as np

from .augmentation import Augmentation
from ....common import Transferable
from ..sample import Sample


class Severity(Transferable, is_base_type=True):
    def __init__(self, severity: int) -> None:
        self.severity = severity


class SeverityAugmentation(Augmentation, Transferable):
    ...
