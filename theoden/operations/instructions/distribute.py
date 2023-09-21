from __future__ import annotations

from abc import ABC, abstractmethod
import random

from ...common import Transferable
from ...topology import TopologyRegister


class Distributor(ABC, Transferable, is_base_type=True):
    """A distributor is used to select nodes from the topology register. This is used to select the nodes that are to be used for an instruction."""

    @abstractmethod
    def select_nodes(self, topology_register: TopologyRegister) -> list[str]:
        pass


class AllDistributor(Distributor, Transferable):
    def select_nodes(self, topology_register: TopologyRegister) -> list[str]:
        return topology_register.get_connected_nodes()


class PercentageDistributor(Distributor, Transferable):
    def __init__(self, percentage: int, seed: int | None = None):
        self.percentage = percentage
        self.seed = seed

    def select_nodes(self, topology_register: TopologyRegister) -> list[str]:
        # set random seed and shuffle the list
        if self.seed is not None:
            random.seed(self.seed)
        node_list = topology_register.get_connected_nodes()
        random.shuffle(node_list)
        return node_list[: int(len(node_list) * self.percentage / 100)]


class NDistributor(Distributor, Transferable):
    def __init__(
        self,
        n: int,
        seed: int | None = None,
        set_flag: str | None = None,
        remove_flag: str | None = None,
    ):
        self.n = n
        self.seed = seed
        self.set_flag = set_flag
        self.remove_flag = remove_flag

    def select_nodes(self, topology_register: TopologyRegister) -> list[str]:
        if self.seed is not None:
            random.seed(self.seed)
        node_list = topology_register.get_connected_nodes()
        nodes = random.sample(node_list, self.n)
        if self.set_flag is not None:
            topology_register.set_flag_of_nodes(nodes, self.set_flag)
        if self.remove_flag is not None:
            topology_register.remove_flag_of_nodes(
                nodes, self.remove_flag, set_flag=False
            )
        return nodes


class FlagDistributor(Distributor, Transferable):
    def __init__(self, flag: str):
        self.flag = flag

    def select_nodes(self, topology_register: TopologyRegister) -> list[str]:
        return topology_register.get_nodes_with_flag(self.flag)


class ListDistributor(Distributor, Transferable):
    def __init__(self, node_list: list[str]):
        self.node_list = node_list

    def select_nodes(self, topology_register: TopologyRegister) -> list[str]:
        return self.node_list
