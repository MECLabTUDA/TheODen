from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..common import Transferable
from .notifications import WatcherNotification


if TYPE_CHECKING:
    from ..topology.server import Server
    from ..topology.node import Node
    from .pool import WatcherPool


class Watcher(Transferable, is_base_type=True):
    """Class for watchers. Watchers are used to monitor different aspects of the framework.

    They do not interfere in the different processes of the framework like executing commands, handling status updates, etc.. For this, please use the corresponding handlers, hooks and functions.
    However, they can still act like saving resources, updating monitoring tools, etc..
    """

    def __init__(
        self,
        notification_of_interest: dict[type[WatcherNotification], callable]
        | None = None,
        fallback_handler: callable | None = None,
    ) -> None:
        """Initialize the watcher

        Args:
            notification_of_interest (dict[type[WatcherNotification], callable]): The notifications of interest
            fallback_handler (callable): The fallback handler
        """
        self.notification_of_interest = notification_of_interest or {}
        self.fallback_handler = fallback_handler

    def set_pool(self, pool: "WatcherPool") -> Watcher:
        """Set the pool of nodes

        Args:
            pool (WatcherPool): The pool of nodes

        Returns:
            Watcher: The watcher itself
        """
        self.pool = pool
        return self

    @property
    def base_topology(self) -> "Server" | "Node":
        """Get the base topology

        Returns:
            Server | Node: The base topology
        """
        return self.pool.base_topology

    def listen(
        self, notification: WatcherNotification, origin: Watcher | None = None
    ) -> None:
        """Function to listen to the pool of nodes

        Args:
            notification (WatcherNotification): The notification
            origin (Watcher): The origin of the notification
        """
        for notification_type, handler in self.notification_of_interest.items():
            if isinstance(notification, notification_type):
                handler(notification, origin)
            else:
                if self.fallback_handler is not None:
                    self.fallback_handler(notification, origin)
