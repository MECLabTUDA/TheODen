from typing import Optional, Any, TypeVar

from ..command import Command
from ....common import Transferable, ExecutionResponse

T = TypeVar("T")


class SetResourceCommand(Command, Transferable):
    def __init__(
        self,
        key: str,
        resource: T,
        overwrite: bool = True,
        assert_type: type[T] = Any,
        *,
        unpack_dict=False,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(uuid=uuid, **kwargs)
        self.key = key
        self.resource: T = resource
        self.assert_type: type[T] = assert_type
        self.overwrite = overwrite
        self.unpack_dict = unpack_dict

    def modify_resource(self, resource: any) -> any:
        return resource

    def execute(self) -> ExecutionResponse | None:
        if not self.unpack_dict:
            self.node.resources.sr(
                key=self.key,
                resource=self.modify_resource(self.resource),
                assert_type=self.assert_type,
                overwrite=self.overwrite,
            )
        else:
            # if resource_manager is a dict and unpack_dict is True, then unpack the dict and register each key-value pair
            assert isinstance(
                self.resource, dict
            ), "if unpack_dict is True, then resource must be a dict"
            for key, value in self.resource.items():
                self.node_rm.sr(
                    key=f"{self.key}:{key}",
                    resource=self.modify_resource(value),
                    assert_type=self.assert_type,
                    overwrite=self.overwrite,
                )
        return None
