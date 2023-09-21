from typing import TYPE_CHECKING, Optional

from .request import ServerRequest
from ...common import Transferable
from ...resources import Loss


if TYPE_CHECKING:
    from theoden.topology.server import Server


class GetResourceCheckpointRequest(ServerRequest, Transferable):
    def __init__(
        self,
        resource_type: str,
        resource_key: str,
        checkpoint_key: str,
        uuid: None | str = None,
        server: Optional["Server"] = None,
        **kwargs,
    ):
        super().__init__(uuid=uuid, server=server, **kwargs)
        self.resource_type = resource_type
        self.resource_key = resource_key
        self.checkpoint_key = checkpoint_key

    def execute(self):
        from ...resources.meta import Checkpoint

        return self.server.resource_register.gr(
            f"__checkpoints__:{self.resource_type}:{self.resource_key}:{self.checkpoint_key}",
            assert_type=Checkpoint,
            default=None,
        ).to(bytes)


class GetBestCheckpointRequest(ServerRequest, Transferable):
    def __init__(
        self,
        resource_type: str = "model",
        resource_key: str = "model",
        dataset_split: str = "val",
        uuid: None | str = None,
        server: Optional["Server"] = None,
        **kwargs,
    ):
        super().__init__(uuid=uuid, server=server, **kwargs)
        self.resource_type = resource_type
        self.dataset_split = dataset_split
        self.resource_key = resource_key

    def execute(self):
        from ...resources.meta import Checkpoint

        self.server.resource_register.gr("losses", assert_type=list[Loss])

        return self.server.resource_register.gr(
            f"__checkpoints__:{self.resource_type}:{self.resource_key}:",
            assert_type=Checkpoint,
            default=None,
        ).to(bytes)
