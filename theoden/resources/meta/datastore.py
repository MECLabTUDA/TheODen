from typing import Any, Union, TypeVar
from ..resource import ResourceRegister

T = TypeVar("T")


class DataSetComponents(ResourceRegister):
    ...


class DataGroup(ResourceRegister):
    ...


class Datastore(ResourceRegister):
    def add_datagroup(self):
        ...

    def sr(self, key: str, resource: T, overwrite: bool = True) -> T:
        return super().sr(key, resource, Union[DataGroup, DataSetComponents], overwrite)
