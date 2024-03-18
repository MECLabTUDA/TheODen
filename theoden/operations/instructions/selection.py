from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod

from ...common import Transferable
from ...topology import Topology
from ..commands import Command


class Selector(ABC, Transferable, is_base_type=True):
    """A distributor is used to select clients from the topology register.
    This is used to select the clients that are to be used for an instruction."""

    @abstractmethod
    def selection(
        self, topology: Topology, commands: list[Command]
    ) -> dict[str, str | None]:
        """Select clients from the topology register.

        Args:
            topology (Topology): The topology register.
            num_commands (int): The number of commands to distribute.

        Returns:
            list[str]: The names of the selected clients.
        """
        raise NotImplementedError(
            "Selector is an abstract class and cannot be instantiated."
        )


class BinarySelector(Selector):
    """Binary Selector are special Selectors if only one command is to be distributed."""

    def select_clients(self, topology: Topology) -> list[str]:
        raise NotImplementedError("Please implement select_clients()")

    def selection(
        self, topology: Topology, commands: list[Command]
    ) -> dict[str, str | None]:
        if len(commands) > 1:
            logging.warning(
                "BinarySelector is not designed for more than one command. Ignoring all but the first command."
            )

        # select clients. clients that are not selected are set to None
        selected_clients = self.select_clients(topology)
        return {
            client: commands[0].uuid if client in selected_clients else None
            for client in topology.online_clients(True)
        }


class AllSelector(BinarySelector):
    """Select all connected clients in the topology register."""

    def select_clients(self, topology: Topology) -> list[str]:
        return topology.online_clients(True)


class PercentageSelector(BinarySelector):
    """Select a percentage of connected clients from the topology register."""

    def __init__(
        self,
        percentage: int,
        seed: int | None = None,
        **kwargs,
    ):
        """Select a percentage of connected clients from the topology register.

        Args:
            percentage (int): The percentage of clients to select.
            seed (int, optional): The random seed to use. Defaults to None.
        """
        super().__init__(**kwargs)
        self.percentage = percentage
        self.seed = seed

    def select_clients(self, topology: Topology) -> list[str]:
        # set random seed and shuffle the list
        if self.seed is not None:
            random.seed(self.seed)
        client_list = topology.online_clients(True)
        random.shuffle(client_list)
        return client_list[: int(len(client_list) * self.percentage / 100)]


class NSelector(BinarySelector):
    """Select n connected clients from the topology register."""

    def __init__(
        self,
        n: int,
        seed: int | None = None,
        **kwargs,
    ):
        """Select random n connected clients from the topology register.

        Args:
            n (int): The number of clients to select.
            seed (int, optional): The random seed to use. Defaults to None.
        """
        super().__init__(**kwargs)
        self.n = n
        self.seed = seed

    def select_clients(self, topology: Topology) -> list[str]:
        if self.seed is not None:
            random.seed(self.seed)
        client_list = topology.online_clients(True)
        clients = random.sample(client_list, self.n)
        return clients


class FlagSelector(BinarySelector):
    """Select all connected clients with a given flag from the topology register."""

    def __init__(
        self,
        flag: str,
        **kwargs,
    ):
        """Select all connected clients with a given flag from the topology register.

        Args:
            flag (str): The flag to select clients with.
            set_flag (str | list[str], optional): The flag to set on the selected clients. Defaults to None.
            remove_flag (str | list[str], optional): The flag to remove from the selected clients. Defaults to None.
        """
        super().__init__(**kwargs)
        self.flag = flag

    def select_clients(self, topology: Topology) -> list[str]:
        return [
            client.name
            for client in topology.get_nodes_with_flag(topology.clients, self.flag)
        ]


class ListSelector(BinarySelector):
    """Select a list of clients from the topology register. If a client is not connected, it is ignored."""

    def __init__(
        self,
        client_list: list[str],
        **kwargs,
    ):
        """Select a list of clients from the topology register. If a client is not connected, it is ignored.

        Args:
            client_list (list[str]): The list of clients to select.
            set_flag (str | list[str], optional): The flag to set on the selected clients. Defaults to None.
            remove_flag (str | list[str], optional): The flag to remove from the selected clients. Defaults to None.
        """
        super().__init__(**kwargs)
        self.client_list = client_list

    def select_clients(self, topology: Topology) -> list[str]:
        return self.client_list & topology.online_clients(True)


class RandomNumberSelector(BinarySelector):
    """Select a random number of connected clients from the topology register."""

    def __init__(
        self,
        seed: int | None = None,
        **kwargs,
    ):
        """Select a random number of connected clients from the topology register.

        Args:
            seed (int, optional): The random seed to use. Defaults to None.
            set_flag (str | list[str], optional): The flag to set on the selected clients. Defaults to None.
            remove_flag (str | list[str], optional): The flag to remove from the selected clients. Defaults to None.
        """
        super().__init__(**kwargs)
        self.seed = seed

    def select_clients(self, topology: Topology) -> list[str]:
        if self.seed is not None:
            random.seed(self.seed)
        return random.sample(
            topology.online_clients(True),
            random.randint(1, len(topology.online_clients(True))),
        )
