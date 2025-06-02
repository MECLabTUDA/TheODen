from __future__ import annotations

import json
from enum import Enum
from typing import Any, Generic, Type, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, validator


from .transferables import Transferables



def is_instance_of_type_hint(obj: any, hint: type):
    """
    Check whether `obj` is an instance of the given type hint.

    Args:
        obj (any): The object to check.
        hint (type): The type hint.

    Returns:
        bool: Whether `obj` is an instance of the given type hint.
    """
    if hasattr(hint, "__origin__"):
        if hint.__origin__ == dict:
            key_hint, value_hint = hint.__args__
            if not isinstance(obj, dict):
                return False
            for key, value in obj.items():
                if not is_instance_of_type_hint(
                    key, key_hint
                ) or not is_instance_of_type_hint(value, value_hint):
                    return False
            return True
        elif hint.__origin__ == list or hint.__origin__ == set:
            (value_hint,) = hint.__args__
            if not isinstance(obj, (list, set)):
                return False
            for item in obj:
                if not is_instance_of_type_hint(item, value_hint):
                    return False
            return True
        elif hint.__origin__ == tuple:
            if not isinstance(obj, tuple):
                return False
            if len(obj) != len(hint.__args__):
                return False
            for i, item_hint in enumerate(hint.__args__):
                if not is_instance_of_type_hint(obj[i], item_hint):
                    return False
            return True
        elif hint.__origin__ == Union:
            for union_type in hint.__args__:
                if is_instance_of_type_hint(obj, union_type):
                    return True
            return False
        else:
            return isinstance(obj, hint.__origin__)
    else:
        return isinstance(obj, hint)


class RegisteredTypeModel(BaseModel):
    datatype: str
    data: dict[str, dict[str, Any]]

    @validator("datatype")
    def is_registered_type(cls, v):
        if v not in Transferables():
            raise ValueError(f"{v} is not a registered type")
        return v


class ExecutionResponse(BaseModel):
    data: dict | None = None
    files: dict[str, bytes] | dict[str, str] | None = None
    response_type: str | None = None

    def contains_files(self) -> bool:
        # check if files are none or empty
        return self.files is not None and self.files

    def get_data(self) -> dict | str:
        # If v is not None, convert it to a json string
        return self.data if self.data else {}

    def get_files(self, as_bytes: bool = True) -> dict[str, bytes] | dict[str, str]:
        """Returns the files as a dictionary of file_name: file_path or file_name: file_bytes

        Args:
            as_bytes (bool): Whether to return the files as bytes or paths. Default is True.

        Returns:
            A dictionary of file_name: file_path or file_name: file_bytes

        Raises:
            ValueError: If files were saved as bytes and as_bytes is False
        """

        return_dict = {}

        if self.files is not None:
            for file_name, file_load in self.files.items():
                # If file_load is a string, it is a path to a file
                if isinstance(file_load, str) and as_bytes:
                    with open(file_load, "rb") as f:
                        return_dict[file_name] = f.read()
                elif isinstance(file_load, bytes) and not as_bytes:
                    raise ValueError(
                        "Files were saved as bytes. Cannot convert to path."
                    )
                else:
                    return_dict[file_name] = file_load
        return return_dict

    def __str__(self):
        files_str = f"{self.files}" 
        if len(files_str) > 100:
            files_str = f"{files_str[:100]}... (truncated)"
        data_str = f"{self.data}"
        if len(data_str) > 100:
            data_str = f"{data_str[:100]}... (truncated)"
        return f"{self.__class__.__name__}(data={data_str}, files={files_str}, response_type={self.response_type})"


class TransmissionExecutionResponse(BaseModel):
    data: dict | None = None
    files: dict[str, str] | None = None
    response_type: str | None = None

    def contains_files(self) -> bool:
        return self.files is not None and self.files


class MetricResponse(ExecutionResponse):
    def __init__(
        self,
        metrics: dict[str, float],
        metric_type: str,
        comm_round: int | None = None,
        epoch: int | None = None,
    ):
        if comm_round is not None:
            if not isinstance(comm_round, int):
                raise ValueError("Comm round must be an integer")
        if epoch is not None:
            if not isinstance(epoch, int):
                raise ValueError("Epoch must be an integer")
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be a dictionary")
        for metric_name, metric_value in metrics.items():
            if not isinstance(metric_name, str):
                raise ValueError("Metric names must be strings")
            if not isinstance(metric_value, (float | int | None)):
                raise ValueError("Metric values must be floats or int")

        super().__init__(
            data={
                "metrics": metrics,
                "comm_round": comm_round,
                "epoch": epoch,
                "metric_type": metric_type,
            },
            files=None,
            response_type="metric",
        )


class ResourceResponse(ExecutionResponse):
    def __init__(self, resource_type: str, resource_manager: dict[str, any]):
        if not isinstance(resource_manager, dict):
            raise ValueError("ResourceManager must be a dictionary")
        for resource_name, _ in resource_manager.items():
            if not isinstance(resource_name, str):
                raise ValueError("Resource names must be strings")

        super().__init__(
            data={"resource_type": resource_type},
            files=resource_manager,
            response_type="resource",
        )


class ClientScoreResponse(ExecutionResponse):
    def __init__(self, score: float | int, score_type: str):
        if not isinstance(score, (float | int)):
            raise ValueError("Score must be a float or int")
        if not isinstance(score_type, str):
            raise ValueError("Score type must be a string")

        super().__init__(
            data={"score": score, "score_type": score_type},
            files=None,
            response_type="score",
        )


class StatusUpdate(BaseModel):
    command_uuid: str
    status: int
    datatype: str
    client_name: str | None = None
    response: ExecutionResponse | None = None

    def contains_files(self) -> bool:
        return self.response is not None and self.response.contains_files()

    def get_response_data(self, as_type: type = dict) -> dict:
        return self.response.get_data() if self.response else {}

    def get_response_files(self) -> dict[str | bytes]:
        return self.response.get_files() if self.response else {}

    def unload(self, files: dict[str, str]) -> TransmissionStatusUpdate:
        return TransmissionStatusUpdate(
            command_uuid=self.command_uuid,
            status=self.status,
            datatype=self.datatype,
            client_name=self.client_name,
            response=TransmissionExecutionResponse(
                data=self.response.data,
                files=files,
                response_type=self.response.response_type,
            )
            if self.response
            else None,
        )
    
    def __str__(self):
        from theoden.operations.status import CommandExecutionStatus
        class_name = self.__class__.__name__
        status_name = CommandExecutionStatus(self.status).name
        return f"{class_name}(command_uuid={self.command_uuid}, status={status_name}, datatype={self.datatype}, client_name={self.client_name}, response={self.response})"


class TransmissionStatusUpdate(StatusUpdate):
    response: TransmissionExecutionResponse | None = None

    def refill(self, files: dict[str, bytes]) -> StatusUpdate:
        return StatusUpdate(
            command_uuid=self.command_uuid,
            status=self.status,
            datatype=self.datatype,
            client_name=self.client_name,
            response=ExecutionResponse(
                data=self.response.data,
                files=files,
                response_type=self.response.response_type,
            )
            if self.response
            else None,
        )
