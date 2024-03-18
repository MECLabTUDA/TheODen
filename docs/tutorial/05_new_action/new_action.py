from theoden.operations import (
    Action,
    Instruction,
    ClosedDistribution,
    ExitRunCommand,
)
from theoden.resources import ResourceManager
from theoden.topology import Topology


class NewAction(Action):
    def __init__(self, **kwargs) -> None:
        super().__init__(None, True, **kwargs)

    def perform(
        self, topology: Topology, resource_manager: ResourceManager
    ) -> Instruction | None:
        print(topology.client_names)
        return None


class NewActionThatExitsAfterwardsAction(Action):
    def __init__(self, **kwargs) -> None:
        super().__init__(None, True, **kwargs)

    def perform(
        self, topology: Topology, resource_manager: ResourceManager
    ) -> Instruction | None:
        print(topology.client_names)
        return ClosedDistribution(ExitRunCommand())
