from pathlib import Path

from ..common import GlobalContext
from ..resources import Loss
from .metric_collector import MetricCollectionWatcher
from .notifications import (
    InitializationNotification,
    MetricNotification,
    NewBestModelNotification,
)
from .watcher import Watcher

import logging
logger = logging.getLogger(__name__)

class BestModelSaverWatcher(Watcher):
    def __init__(
        self,
        listen_to: str | None = None,
        save_folder: str | None = None,
        model_key: str | list[str] | None = None,
    ) -> None:
        super().__init__(
            notification_of_interest={
                NewBestModelNotification: self._handle,
                InitializationNotification: self._set_run_name,
            }
        )
        self.listen_to = listen_to
        self.save_folder = save_folder
        self.model_key = model_key
        self.run_name = ""

    def _set_run_name(
        self, notification: InitializationNotification, origin: Watcher | None = None
    ) -> None:
        self.run_name = notification.run_name
        self.save_folder = (
            self.save_folder
            if self.save_folder is not None
            else GlobalContext()["model_save_folder"]
        )

    def _handle(
        self, notification: NewBestModelNotification, origin: Watcher | None = None
    ) -> None:
        if self.listen_to is None:
            losses = self.pool.base_topology.resources.gr(
                "losses", assert_type=list[Loss]
            )
            self.listen_to = Loss.get_choosing_criterion(losses).display_name()

        if notification.metric == self.listen_to:
            cm = self.base_topology.resources.checkpoint_manager

            path = (
                Path(self.save_folder)
                / self.run_name
                / f"{self.model_key}_best_{notification.split}.pt"
                if self.run_name
                else Path(self.save_folder)
                / f"{self.model_key}_best_{notification.split}.pt"
            )

            logger.info(
                f"Saving new best {notification.split} model '{self.model_key}' as '{self.model_key}_best_{notification.split}' to '{path.as_posix()}'"
            )

            path.parent.mkdir(parents=True, exist_ok=True)

            cm.copy_checkpoint(
                resource_type="model",
                resource_key=self.model_key,
                checkpoint_key="__global__",
                new_checkpoint_key=f"{self.model_key}_best_{notification.split}",
            ).save(path=path)


class SaveEveryNRoundWatcher(MetricCollectionWatcher):
    def __init__(
        self,
        n_round: int,
        split="train",
        save_folder: str | None = None,
        model_key: str = "model",
    ) -> None:
        super().__init__(
            notification_of_interest={
                InitializationNotification: self._set_run_name,
            })
        self.n_round = n_round
        self.split = split
        self.save_folder = save_folder
        self.model_key = model_key
        self.run_name = ""

    def _set_run_name(
        self, notification: InitializationNotification, origin: Watcher | None = None
    ) -> None:
        self.run_name = notification.run_name
        self.save_folder = self.save_folder or GlobalContext()["model_save_folder"]

    def _handle_metric(
        self, notification: MetricNotification, origin: Watcher | None = None
    ) -> None:
        if notification.comm_round % self.n_round == 0:
            cm = self.base_topology.resources.checkpoint_manager

            if self.run_name:
                path = (
                    Path(self.save_folder)
                    / self.run_name
                    / f"{self.model_key}_round_{notification.comm_round}.pt"
                )
            else:
                path = Path(self.save_folder) / f"{self.model_key}_round_{notification.comm_round}.pt"


            logger.info(
                f"Saving model '{self.model_key}' at round {notification.comm_round} to '{path.as_posix()}'"
            )

            path.parent.mkdir(parents=True, exist_ok=True)


            
            cm.get_checkpoint(
                resource_type="model",
                resource_key=self.model_key,
                checkpoint_key="__global__"
            ).save(path=path)

            # Why would you want to copy the checkpoint?
            # This would flood the checkpoint manager and hence the GPU memory.
            #cm.copy_checkpoint(
            #    resource_type="model",
            #    resource_key=self.model_key,
            #    checkpoint_key="__global__",
            #    new_checkpoint_key=f"{self.model_key}_round_{notification.comm_round}",
            #).save(path=path)
