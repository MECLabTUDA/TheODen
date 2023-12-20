from __future__ import annotations

import io
from copy import deepcopy

import torch

from ...operations.commands import (
    Command,
    LoadOptimizerStateDictCommand,
    LoadStateDictCommand,
)
from ..resource import ResourceManager


class Checkpoint:
    def to(self, datatype: type) -> dict | bytes:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def save(
        self, path: str, return_as_file_checkpoint: bool = True
    ) -> FileCheckpoint | Checkpoint:
        with open(path, "wb") as f:
            if isinstance(self.data, dict):
                torch.save(self.data, f)
            else:
                f.write(self.data)
        if return_as_file_checkpoint:
            return FileCheckpoint(path=path, base_type=type(self))
        else:
            return self


class FileCheckpoint(Checkpoint):
    def __init__(self, path: str, base_type: type) -> None:
        self.path = path
        self.base_type = base_type

    def to(self, datatype: type) -> dict | bytes:
        if datatype == dict:
            return torch.load(self.path)
        elif datatype == bytes:
            with open(self.path, "rb") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported datatype {datatype}")


class DictCheckpoint(Checkpoint):
    def __init__(self, state_dict: dict) -> None:
        self.data = state_dict

    def to(self, datatype: type) -> dict | bytes:
        if datatype == dict:
            return self.data
        elif datatype == bytes:
            bytes_io = io.BytesIO()
            torch.save(self.data, bytes_io)
            return bytes_io.getvalue()
        else:
            raise ValueError(f"Unsupported datatype {datatype}")


class BytesCheckpoint(Checkpoint):
    def __init__(self, data_bytes: bytes) -> None:
        self.data = data_bytes

    def to(self, datatype: type) -> dict | bytes:
        if datatype == dict:
            bytes_io = io.BytesIO(self.data)
            return torch.load(bytes_io)
        elif datatype == bytes:
            return self.data
        else:
            raise ValueError(f"Unsupported datatype {datatype}")


class ErrorCheckpoint(Checkpoint):
    def __init__(self, error: str) -> None:
        self.error = error

    def to(self, datatype: type) -> dict | bytes:
        raise RuntimeError(self.error)


class Checkpoints(ResourceManager):
    def register_checkpoint(self, checkpoint_key: str, checkpoint: Checkpoint) -> None:
        self.sr(key=checkpoint_key, resource=checkpoint, assert_type=Checkpoint)


class TypedCheckpoints(ResourceManager[str, Checkpoints]):
    default_subregister_type = Checkpoints

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def register_checkpoint(
        self, resource_key: str, checkpoint_key: str, checkpoint: Checkpoint
    ) -> None:
        cps = self.gr(resource_key, assert_type=Checkpoints, default=None)

        if cps is None:
            self.sr(resource_key, Checkpoints())
            cps = self.gr(resource_key, assert_type=Checkpoints)

        cps.register_checkpoint(checkpoint_key=checkpoint_key, checkpoint=checkpoint)

    def get_loading_command(
        self, resource_type: str, resource_key: str, checkpoint_key: str
    ) -> Command:
        raise NotImplementedError("This method should be implemented by the subclass.")


class ModelCheckpoints(TypedCheckpoints):
    def get_loading_command(
        self, resource_type: str, resource_key: str, checkpoint_key: str
    ) -> Command:
        return LoadStateDictCommand(
            resource_type=resource_type,
            resource_key=resource_key,
        )


class OptimizerCheckpoints(TypedCheckpoints):
    def get_loading_command(
        self, resource_type: str, resource_key: str, checkpoint_key: str
    ) -> Command:
        return LoadOptimizerStateDictCommand(
            resource_key=resource_key,
            checkpoint_key=checkpoint_key,
        )


class CheckpointManager(ResourceManager):
    default_subregister_type = TypedCheckpoints

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().clear()
        self["model"] = ModelCheckpoints()
        self["optimizer"] = OptimizerCheckpoints()

    def register_checkpoint(
        self,
        resource_type: str,
        resource_key: str,
        checkpoint_key: str,
        checkpoint: Checkpoint,
        create_type_if_not_exists: type[TypedCheckpoints] = None,
    ):
        typed_cp = self.gr(
            key=resource_type, assert_type=TypedCheckpoints, default=None
        )

        if typed_cp is None:
            if create_type_if_not_exists is None:
                raise ValueError(
                    f"Resource type {resource_type} does not exist and no type was specified to create it."
                )

            self.sr(resource_type, create_type_if_not_exists())
            typed_cp = self.gr(resource_type, assert_type=TypedCheckpoints)

        typed_cp.register_checkpoint(
            resource_key=resource_key,
            checkpoint_key=checkpoint_key,
            checkpoint=checkpoint,
        )

    def get_global_checkpoints_commands(
        self, of_resource_type: list[str] | None = None
    ) -> list[Command]:
        cmds = []
        for resource_type, typed_cp in self.items():
            if isinstance(typed_cp, TypedCheckpoints) and (
                of_resource_type is None or resource_type in of_resource_type
            ):
                for resource_key, cps in typed_cp.items():
                    if isinstance(cps, Checkpoints):
                        for checkpoint_key, cp in cps.items():
                            if checkpoint_key == "__global__":
                                cmds.append(
                                    typed_cp.get_loading_command(
                                        resource_type=resource_type,
                                        resource_key=resource_key,
                                        checkpoint_key=checkpoint_key,
                                    )
                                )
        return cmds

    def get_checkpoint(
        self,
        resource_type: str,
        resource_key: str,
        checkpoint_key: str,
    ) -> Checkpoint:
        typed_cp = self.gr(
            key=resource_type, assert_type=TypedCheckpoints, default=None
        )

        if typed_cp is None:
            raise KeyError(f"Resource type {resource_type} does not exist.")

        cps = typed_cp.gr(key=resource_key, assert_type=Checkpoints, default=None)

        if cps is None:
            raise KeyError(f"Resource {resource_key} does not exist.")

        cp = cps.gr(key=checkpoint_key, assert_type=Checkpoint, default=None)

        if cp is None:
            raise KeyError(f"Checkpoint {checkpoint_key} does not exist.")

        return cp

    def copy_checkpoint(
        self,
        resource_type: str,
        resource_key: str,
        checkpoint_key: str,
        new_checkpoint_key: str,
    ) -> Checkpoint:
        cp = self.get_checkpoint(
            resource_type=resource_type,
            resource_key=resource_key,
            checkpoint_key=checkpoint_key,
        )
        copied_cp = deepcopy(cp)
        self.register_checkpoint(
            resource_type=resource_type,
            resource_key=resource_key,
            checkpoint_key=new_checkpoint_key,
            checkpoint=copied_cp,
        )
        return copied_cp
