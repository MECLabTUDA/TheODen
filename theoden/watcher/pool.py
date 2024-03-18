from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .notifications import WatcherNotification
from .watcher import Watcher

if TYPE_CHECKING:
    from ..topology.client import Client
    from ..topology.server import Server


class WatcherPool:
    def __init__(self, base_topology: "Server" | "Client") -> None:
        """A pool of watchers

        Args:
            base_topology (Server | Node): The base topology of the watcher pool
        """

        self.watchers: set[Watcher] = set()
        self.base_topology = base_topology

    def add(self, watcher: Watcher | list[Watcher]) -> WatcherPool:
        """Add a watcher/multiple watcher to the pool and set the pool of the watcher(s) to this pool

        Args:
            watcher (Watcher | list[Watcher]): The watcher(s) to add

        Returns:
            WatcherPool: The watcher pool
        """
        if isinstance(watcher, list):
            for w in watcher:
                self.watchers.add(w.set_pool(self))
        else:
            self.watchers.add(watcher.set_pool(self))
        return self

    def remove(self, watcher: Watcher) -> WatcherPool:
        """Remove a watcher from the pool

        Args:
            watcher (Watcher): The watcher to remove
        """
        self.watchers.remove(watcher)
        return self

    def notify_all(
        self, notification: WatcherNotification, origin: Watcher | None = None
    ) -> WatcherPool:
        """Notify all watchers in the pool

        Args:
            notification (WatcherNotification): The notification to send

        Returns:
            WatcherPool: The watcher pool
        """
        for watcher in self.watchers:
            try:
                watcher.listen(notification, origin=origin)
            except Exception as e:
                logging.warning(
                    f"Exception while notifying watcher {type(watcher).__name__}: {e}"
                )
        return self

    def notify_of_type(
        self,
        notification: WatcherNotification,
        of_type: type[Watcher],
        origin: Watcher | None = None,
    ) -> WatcherPool:
        """Notify all watchers of a specific type in the pool

        Args:
            notification (WatcherNotification): The notification to send
            type (type[Watcher]): The type of watcher to notify

        Returns:
            WatcherPool: The watcher pool
        """
        for watcher in self.watchers:
            if isinstance(watcher, of_type):
                try:
                    watcher.listen(notification, origin=origin)
                except Exception as e:
                    logging.warning(
                        f"Exception while notifying watcher {type(watcher).__name__}: {e}"
                    )
        return self
