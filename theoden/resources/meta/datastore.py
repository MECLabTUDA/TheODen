from typing import Any, Union, TypeVar
from ..resource import ResourceManager

T = TypeVar("T")


class DataSetComponents(ResourceManager):
    ...


class DataGroup(ResourceManager):
    ...


class Datastore(ResourceManager):
    def add_datagroup(self):
        ...

    def sr(self, key: str, resource: T, overwrite: bool = True) -> T:
        return super().sr(key, resource, Union[DataGroup, DataSetComponents], overwrite)
