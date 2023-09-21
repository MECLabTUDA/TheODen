from pathlib import Path

from ..common import Transferable, GlobalContext
from .watcher import Watcher
from .notifications import (
    NewBestModelNotification,
    InitializationNotification,
)
from ..resources import Loss


class ModelSaverWatcher(Watcher, Transferable):
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
        self.save_folder = (
            save_folder
            if save_folder is not None
            else GlobalContext()["model_save_folder"]
        )
        self.model_key = model_key
        self.run_name = ""

    def _set_run_name(
        self, notification: InitializationNotification, origin: Watcher | None = None
    ) -> None:
        self.run_name = notification.run_name

    def _handle(
        self, notification: NewBestModelNotification, origin: Watcher | None = None
    ) -> None:
        if self.listen_to is None:
            losses = self.pool.base_topology.resource_register.gr(
                "losses", assert_type=list[Loss]
            )
            self.listen_to = Loss.get_choosing_criterion(losses).display_name()

        if notification.metric == self.listen_to:
            cm = self.base_topology.resource_register.checkpoint_manager

            path = (
                Path(self.save_folder)
                / self.run_name
                / f"{self.model_key}_best_{notification.split}.pt"
                if self.run_name
                else Path(self.save_folder)
                / f"{self.model_key}_best_{notification.split}.pt"
            )

            print(
                f"Saving new best {notification.split} model '{self.model_key}' as '{self.model_key}_best_{notification.split}' to '{path.as_posix()}'"
            )

            path.parent.mkdir(parents=True, exist_ok=True)

            cm.copy_checkpoint(
                resource_type="model",
                resource_key=self.model_key,
                checkpoint_key="__global__",
                new_checkpoint_key=f"{self.model_key}_best_{notification.split}",
            ).save(path=path)
