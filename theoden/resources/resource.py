from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Type, TypeVar

from ..common.typing import is_instance_of_type_hint
from ..networking import FileStorageInterface

if TYPE_CHECKING:
    from ..watcher import WatcherPool
    from .meta import CheckpointManager


T = TypeVar("T")
D = TypeVar("D")


class ResourceManager(OrderedDict):
    """A manager for resources

    The resource register is a dictionary that can be used to register resources.
    ResourceManager can be retrieved by their key. If the key contains a ':', the resource will be retrieved from a subregister.

    Examples:
        >>> from theoden.resources import ResourceManager

        >>> class MyResource:
        ...     def __init__(self, name: str):
        ...         self.name = name

        >>> resource_manager = ResourceManager()
        >>> resource_manager.sr(key="my_resource", resource=MyResource(name="my_resource"))

        >>> resource_manager.gr(key="my_resource", assert_type=MyResource)
        <MyResource name="my_resource">

        >>> resource_manager.gr(key="my_resource", assert_type=int)
        AssertionError: Resource not of type `int` but of type `MyResource`

        >>> resource_manager.gr_of_type(MyResource)
        {'my_resource': <MyResource name="my_resource">}
    """

    default_subregister_type = None

    def sr(
        self,
        key: str,
        resource: T,
        assert_type: Type[T] = Any,
        overwrite: bool = True,
        return_resource: bool = True,
    ) -> T | None:
        """Register a resource in the resource register

        Args:
            key (str): The key of the resource. If the key contains a ':', the resource will be registered in a subregister.
            resource (any): The resource to register.
            assert_type (type[Resource], optional): The type of the resource. Defaults to None.
            overwrite (bool, optional): Whether to overwrite an existing resource with the same key. Defaults to True.
            return_resource (bool, optional): Whether to return the resource. Defaults to True.

        Raises:
            KeyError: If the resource with the same key already exists and overwrite is False.
            AssertionError: If the resource is not of the specified type.

        Returns:
            any: The registered resource.

        """

        # split by ':'
        splitted_key = key.split(":")
        if len(splitted_key) > 1:
            # check if first key is in resource_manager
            if splitted_key[0] not in self:
                # create new resource register
                sub_register = self.sr(
                    key=splitted_key[0],
                    resource=ResourceManager()
                    if self.default_subregister_type is None
                    else self.default_subregister_type(),
                    assert_type=ResourceManager,
                    overwrite=overwrite,
                )

            else:
                if overwrite:
                    if not isinstance(self[splitted_key[0]], ResourceManager):
                        # set to resource register
                        self[splitted_key[0]] = (
                            ResourceManager()
                            if self.default_subregister_type is None
                            else self.default_subregister_type()
                        )
                # get resource register
                sub_register = self.gr(key=splitted_key[0], assert_type=ResourceManager)

            # register resource in subregister with all except the first key
            return sub_register.sr(
                key=":".join(splitted_key[1:]),
                resource=resource,
                assert_type=assert_type,
                overwrite=overwrite,
                return_resource=return_resource,
            )

        if assert_type is not Any:
            assert is_instance_of_type_hint(
                resource, assert_type
            ), f"Resource not of type `{assert_type if hasattr(assert_type, '__origin__') else assert_type.__name__}` but of type `{type(resource).__name__}`"
        if key in self and not overwrite:
            raise KeyError(f"Resource with key `{key}` already exists")
        self[key] = resource
        return resource if return_resource else None

    def gr(
        self,
        key: str,
        assert_type: Type[T] = Any,
        default: D = ...,
    ) -> T | D:
        """Load a resource from the resource register

        Args:
            key (str): The key of the resource. If the key contains a ':', the resource will be retrieved from a subregister.
            assert_type (type[Resource], optional): The type of the resource. Defaults to None.
            default (any, optional): The default value to return if the resource does not exist. Defaults to ... (no default).

        Raises:
            KeyError: If the resource with the same key already exists and overwrite is False.
            AssertionError: If the resource is not of the specified type.

        Returns:
            any: The registered resource.
        """

        # split by ':'
        split_key = key.split(":")
        if len(split_key) > 1:
            # get resource register
            sub_register = self.gr(key=split_key[0], assert_type=ResourceManager)
            # get resource from subregister with all except the first key
            return sub_register.gr(
                key=":".join(split_key[1:]), assert_type=assert_type, default=default
            )

        try:
            _resource = self[key]
        except KeyError:
            # only raise error if default is not ..., otherwise return default
            if default is ...:
                raise KeyError(f"Resource with key `{key}` does not exist")
            else:
                return default

        if assert_type is not Any:
            assert is_instance_of_type_hint(
                _resource, assert_type
            ), f"Resource not of type `{assert_type if hasattr(assert_type, '__origin__') else assert_type.__name__}` but of type `{type(_resource).__name__}`"

        return _resource

    def rm(self, key: str, assert_type: Type[T] = Any) -> T:
        """Remove a resource from the resource register

        Args:
            key (str): The key of the resource. If the key contains a ':', the resource will be removed from a subregister.
            assert_type (type[Resource], optional): The type of the resource. Defaults to None.

        Raises:
            KeyError: If the resource with the same key already exists and overwrite is False.
            AssertionError: If the resource is not of the specified type.

        Returns:
            any: The registered resource.
        """

        # split by ':'
        splitted_key = key.split(":")
        if len(splitted_key) > 1:
            # get resource register
            sub_register = self.gr(key=splitted_key[0], assert_type=ResourceManager)
            # get resource from subregister with all except the first key
            return sub_register.rm(
                key=":".join(splitted_key[1:]), assert_type=assert_type
            )

        _resource = self[key]
        if assert_type is not Any:
            assert is_instance_of_type_hint(
                _resource, assert_type
            ), f"Resource not of type `{assert_type if hasattr(assert_type, '__origin__') else assert_type.__name__}` but of type `{type(_resource).__name__}`"

        return self.pop(key)

    def __contains__(self, key: str) -> bool:
        """Check if a resource is in the resource register

        Args:
            key (str): The key of the resource. If the key contains a ':', the resource will be checked in a subregister.

        Returns:
            bool: Whether the resource is in the resource register.
        """

        # split by ':'
        splitted_key = key.split(":")
        if len(splitted_key) > 1:
            # get resource register
            sub_register = self.gr(splitted_key[0], assert_type=ResourceManager)
            # get resource from subregister with all except the first key
            return sub_register.__contains__(key=":".join(splitted_key[1:]))

        return super().__contains__(key)

    def cp(
        self,
        origin_key: str,
        destiny_key: str,
        assert_type: Type[T] = Any,
        overwrite: bool = True,
        default: any = ...,
    ) -> None:
        """Copy a resource from one key to another

        Args:
            origin_key (str): The key of the resource to copy
            destiny_key (str): The key of the resource to copy to
            assert_type (type, optional): The type of the resource. Defaults to Any.
            overwrite (bool, optional): Whether to overwrite the resource if it already exists. Defaults to True.
            default (any, optional): The default value to return if the resource does not exist. Defaults to ... (no default).

        Raises:
            KeyError: If the resource with the same key already exists and overwrite is False.
            AssertionError: If the resource is not of the specified type.
        """
        self.sr(
            key=destiny_key,
            resource=self.gr(origin_key, default=default),
            assert_type=assert_type,
            overwrite=overwrite,
        )

    def gr_of_type(self, resource_type: T) -> dict[str, T]:
        """Get all resource_manager of a given type

        Args:
            resource_type (type): The type of the resource_manager to get

        Returns:
            dict[str, any]: The resource_manager of the given type
        """

        # use resource_manager ans subregister

        resource_manager = {}
        for key, resource in self.items():
            if is_instance_of_type_hint(resource, resource_type):
                resource_manager[key] = resource
            elif isinstance(resource, ResourceManager):
                sub_resource_manager = resource.gr_of_type(resource_type)
                for sub_key, sub_resource in sub_resource_manager.items():
                    resource_manager[f"{key}:{sub_key}"] = sub_resource
        return resource_manager

    """ Helper functions """

    def _recursive_unpack(self, d: dict) -> dict:
        new_d = {}
        for key, val in d.items():
            if isinstance(val, ResourceManager):
                new_d[key] = self._recursive_unpack(val)
            else:
                new_d[key] = type(val).__name__
        return new_d

    def get_key_type_dict(self) -> dict[str, str]:
        return self._recursive_unpack(self)

    @property
    def watcher(self) -> "WatcherPool":
        """Returns the watcher pool and creates it if it does not exist

        Returns:
            WatcherPool: The watcher pool
        """
        from ..watcher import WatcherPool

        wp = self.gr(key="__watcher__", assert_type=WatcherPool, default=None)
        if wp is None:
            wp = WatcherPool()
            self.sr(key="__watcher__", resource=wp)
        return wp

    @property
    def checkpoint_manager(self) -> "CheckpointManager":
        """Returns the checkpoint manager and creates it if it does not exist

        Returns:
            CheckpointManager: The checkpoint manager
        """

        from .meta import CheckpointManager

        cm = self.gr(key="__checkpoints__", assert_type=CheckpointManager, default=None)
        if cm is None:
            cm = CheckpointManager()
            self.sr(key="__checkpoints__", resource=cm)
        return cm

    @property
    def client_checkpoints(self) -> "CheckpointManager":
        """Returns the checkpoint manager and creates it if it does not exist

        Returns:
            CheckpointManager: The checkpoint manager
        """

        from .meta import CheckpointManager

        cm = self.gr(
            key="__client_checkpoints__", assert_type=CheckpointManager, default=None
        )
        if cm is None:
            cm = CheckpointManager()
            self.sr(key="__client_checkpoints__", resource=cm)
        return cm

    @property
    def storage(self) -> FileStorageInterface:
        """Returns the file storage and creates it if it does not exist

        Returns:
            FileStorage: The file storage
        """

        fs = self.gr(key="__storage__", assert_type=FileStorageInterface, default=None)
        if fs is None:
            raise ValueError("No file storage interface found")
        return fs
