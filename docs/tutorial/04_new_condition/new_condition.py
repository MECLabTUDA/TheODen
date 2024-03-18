from theoden.operations import Condition
from theoden.resources import ResourceManager
from theoden.topology import Topology

import datetime


class LastDigitOfSecondsIsNumberCondition(Condition):
    def __init__(self, number: int):
        super().__init__()
        self.number = number

    def resolved(
        self, resource_manager: ResourceManager, topology: Topology | None = None
    ) -> bool:
        return datetime.datetime.now().second % 10 == self.number
