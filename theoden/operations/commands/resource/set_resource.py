from typing import Any, TypeVar

from ....common import ExecutionResponse
from ..command import Command

T = TypeVar("T")


class SetResourceCommand(Command):
    def __init__(
        self,
        key: str,
        resource: T,
        overwrite: bool = True,
        assert_type: type[T] = Any,
        *,
        unpack_dict=False,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Set a resource on the client

        Args:
            key (str): The resource key
            resource (T): The resource to set
            overwrite (bool, optional): Whether to overwrite the existing resource. Defaults to True.
            assert_type (type[T], optional): The type of the resource. Defaults to Any.
            unpack_dict (bool, optional): Whether to unpack the resource if it is a dict. Defaults to False.
            uuid (str, optional): The uuid of the command. Defaults to None.
        """
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
            self.client.resources.sr(
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
                self.client_rm.sr(
                    key=f"{self.key}:{key}",
                    resource=self.modify_resource(value),
                    assert_type=self.assert_type,
                    overwrite=self.overwrite,
                )
        return None
