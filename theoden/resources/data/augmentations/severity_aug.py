import numpy as np

from ....common import Transferable
from ..sample import Sample
from .augmentation import Augmentation


class Severity(Transferable, is_base_type=True):
    def __init__(self, severity: int) -> None:
        self.severity = severity


class SeverityAugmentation(Augmentation, Transferable):
    ...
