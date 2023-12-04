from __future__ import annotations

from abc import ABC, abstractmethod
import random
import logging

from ...common import Transferable
from ...topology import Topology
from ..commands import Command


class Selector(ABC, Transferable, is_base_type=True):
    """A distributor is used to select nodes from the topology register.
    This is used to select the nodes that are to be used for an instruction."""

    @abstractmethod
    def selection(
        self, topology: Topology, commands: list[Command]
    ) -> dict[str, str | None]:
        """Select nodes from the topology register.

        Args:
            topology (Topology): The topology register.
            num_commands (int): The number of commands to distribute.

        Returns:
            list[str]: The names of the selected nodes.
        """
        raise NotImplementedError(
            "Selector is an abstract class and cannot be instantiated."
        )


class BinarySelector(Selector, Transferable):
    """Binary Selector are special Selectors if only one command is to be distributed."""

    def select_nodes(self, topology: Topology) -> list[str]:
        raise NotImplementedError("Please implement select_nodes()")

    def selection(
        self, topology: Topology, commands: list[Command]
    ) -> dict[str, str | None]:
        if len(commands) > 1:
            logging.warning(
                "BinarySelector is not designed for more than one command. Ignoring all but the first command."
            )

        # select nodes. nodes that are not selected are set to None
        selected_nodes = self.select_nodes(topology)
        return {
            node: commands[0].uuid if node in selected_nodes else None
            for node in topology.online_clients(True)
        }


class AllSelector(BinarySelector, Transferable):
    """Select all connected nodes in the topology register."""

    def select_nodes(self, topology: Topology) -> list[str]:
        return topology.online_clients(True)


class PercentageSelector(BinarySelector, Transferable):
    """Select a percentage of connected nodes from the topology register."""

    def __init__(
        self,
        percentage: int,
        seed: int | None = None,
        **kwargs,
    ):
        """Select a percentage of connected nodes from the topology register.

        Args:
            percentage (int): The percentage of nodes to select.
            seed (int, optional): The random seed to use. Defaults to None.
        """
        super().__init__(**kwargs)
        self.percentage = percentage
        self.seed = seed

    def select_nodes(self, topology: Topology) -> list[str]:
        # set random seed and shuffle the list
        if self.seed is not None:
            random.seed(self.seed)
        node_list = topology.online_clients(True)
        random.shuffle(node_list)
        return node_list[: int(len(node_list) * self.percentage / 100)]


class NSelector(BinarySelector, Transferable):
    """Select n connected nodes from the topology register."""

    def __init__(
        self,
        n: int,
        seed: int | None = None,
        **kwargs,
    ):
        """Select random n connected nodes from the topology register.

        Args:
            n (int): The number of nodes to select.
            seed (int, optional): The random seed to use. Defaults to None.
        """
        super().__init__(**kwargs)
        self.n = n
        self.seed = seed

    def select_nodes(self, topology: Topology) -> list[str]:
        if self.seed is not None:
            random.seed(self.seed)
        node_list = topology.online_clients(True)
        nodes = random.sample(node_list, self.n)
        return nodes


class FlagSelector(BinarySelector, Transferable):
    """Select all connected nodes with a given flag from the topology register."""

    def __init__(
        self,
        flag: str,
        **kwargs,
    ):
        """Select all connected nodes with a given flag from the topology register.

        Args:
            flag (str): The flag to select nodes with.
            set_flag (str | list[str], optional): The flag to set on the selected nodes. Defaults to None.
            remove_flag (str | list[str], optional): The flag to remove from the selected nodes. Defaults to None.
        """
        super().__init__(**kwargs)
        self.flag = flag

    def select_nodes(self, topology: Topology) -> list[str]:
        return [
            node.name
            for node in topology.get_nodes_with_flag(topology.clients, self.flag)
        ]


class ListSelector(BinarySelector, Transferable):
    """Select a list of nodes from the topology register. If a node is not connected, it is ignored."""

    def __init__(
        self,
        node_list: list[str],
        **kwargs,
    ):
        """Select a list of nodes from the topology register. If a node is not connected, it is ignored.

        Args:
            node_list (list[str]): The list of nodes to select.
            set_flag (str | list[str], optional): The flag to set on the selected nodes. Defaults to None.
            remove_flag (str | list[str], optional): The flag to remove from the selected nodes. Defaults to None.
        """
        super().__init__(**kwargs)
        self.node_list = node_list

    def select_nodes(self, topology: Topology) -> list[str]:
        return self.node_list & topology.online_clients(True)


class RandomNumberSelector(BinarySelector, Transferable):
    """Select a random number of connected nodes from the topology register."""

    def __init__(
        self,
        seed: int | None = None,
        **kwargs,
    ):
        """Select a random number of connected nodes from the topology register.

        Args:
            seed (int, optional): The random seed to use. Defaults to None.
            set_flag (str | list[str], optional): The flag to set on the selected nodes. Defaults to None.
            remove_flag (str | list[str], optional): The flag to remove from the selected nodes. Defaults to None.
        """
        super().__init__(**kwargs)
        self.seed = seed

    def select_nodes(self, topology: Topology) -> list[str]:
        if self.seed is not None:
            random.seed(self.seed)
        return random.sample(
            topology.online_clients(True),
            random.randint(1, len(topology.online_clients(True))),
        )
