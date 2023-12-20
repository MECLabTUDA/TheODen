from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

import fastapi
import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from ..common import (
    ExecutionResponse,
    RegisteredTypeModel,
    ServerRequestError,
    StatusUpdate,
    Transferables,
    TransmissionExecutionResponse,
    TransmissionStatusUpdate,
    UnauthorizedError,
)
from ..operations import *
from ..security import create_access_token, decode_token
from ..security.auth import AuthenticationManager, UserRole
from ..topology.topology import Node, NodeStatus, NodeType, Topology
from .interface import NodeInterface
from .storage import FileStorageInterface

if TYPE_CHECKING:
    from ..topology.server import Server


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)


class RestServerInterface(FastAPI):
    def __init__(
        self,
        server: "Server",
        node_config: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.server = server

        self.auth_manager = AuthenticationManager(
            yaml_file=node_config, simulation=node_config is None
        )

        @self.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            request: Request, exc: RequestValidationError
        ):
            # Extract validation errors and details
            validation_errors = exc.errors()
            error_messages = []
            for error in validation_errors:
                field = error["loc"][0] if error["loc"] else "body"
                error_message = f"Field '{field}' {error['msg']}"
                error_messages.append(error_message)

            # Combine error messages into a single string
            error_message_str = "\n".join(error_messages)

            # Log the error message
            logging.error(f"{request}: {error_message_str}")

            # Create a response with more details
            content = {
                "status_code": 10422,
                "message": "Request validation error",
                "errors": error_messages,
            }
            return JSONResponse(
                content=content,
                status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        @self.post("/token")
        async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
            try:
                user = self.auth_manager.authenticate(
                    form_data.username, form_data.password
                )
            except UnauthorizedError:
                raise HTTPException(
                    status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # create a new access token
            access_token = create_access_token(data={"sub": user.username})
            # store the access token in the auth manager
            user.token = access_token

            if (
                self.auth_manager.simulation
                and user.username not in self.server.topology.nodes
            ):
                self.server.topology.add_node(
                    Node(
                        node_name=user.username,
                        node_type=NodeType.CLIENT,
                        status=NodeStatus.ONLINE,
                    )
                )

            self.server.topology.set_online(user.username)

            return {"access_token": access_token, "token_type": "bearer"}

        def authenticate_token(token: str) -> str:
            username = decode_token(token)
            if not self.auth_manager.get_user_by_name(username):
                raise HTTPException(
                    status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return username

        @self.post("/status")
        async def status(
            token: Annotated[str, Depends(oauth2_scheme)],
            status_update: TransmissionStatusUpdate,
        ):
            # authenticate the user
            node_name = authenticate_token(token)

            # add the node uuid to the status update
            status_update.node_name = node_name

            downloaded_resource_manager = {}

            if status_update.contains_files():
                for file_name, file_uuid in status_update.response.files.items():
                    response = self.storage_interface.load_resource(file_uuid)
                    downloaded_resource_manager[file_name] = response

                    # remove file from storage
                    self.storage_interface.remove_resource(file_uuid)

            status_update = status_update.refill(downloaded_resource_manager)

            return self.server.process_status_update(status_update)

        @self.post("/serverrequest")
        async def server_request(
            token: Annotated[str, Depends(oauth2_scheme)],
            request_body: RegisteredTypeModel,
        ):
            # authenticate the user
            node_name = authenticate_token(token)

            # convert the request body to a ServerRequest
            sr = Transferables().to_object(
                request_body.dict(),
                ServerRequest,
                node_name=node_name,
            )

            try:
                # process the server request on the server
                response = self.server.process_server_request(sr)
                files = {}
                if response.contains_files():
                    files = self.storage_interface.upload_resources(
                        response.get_files()
                    )
                response = TransmissionExecutionResponse(
                    data=response.data,
                    files=files,
                    response_type=response.response_type,
                )

            except UnauthorizedError as e:
                raise HTTPException(
                    status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                    detail=str(e),
                )

            return JSONResponse(content=response.dict())

    @property
    def top_reg(self) -> Topology:
        return self.server.topology

    def start(self):
        pass

    def init(self):
        pass

    def add_storage_interface(self, storage_interface: FileStorageInterface) -> None:
        self.storage_interface = storage_interface
        if self.storage_interface.storage is not None:
            self.mount("/", self.storage_interface.storage)


class RestNodeInterface(NodeInterface):
    def __init__(
        self,
        command_queue: list[dict],
        address: str = "localhost",
        port: int = 8000,
        https: bool = False,
        username: str = "dummy",
        password: str = "dummy",
        ping_interval: float = 1.0,
    ) -> None:
        """A federated learning node communication interface based on REST.

        Args:
            address (str, optional): The address of the server. Defaults to "localhost".
            port (int, optional): The port of the server. Defaults to 8000.
            https (bool, optional): Whether to use https. Defaults to False.
            username (str, optional): The username to use for authentication. Defaults to "dummy".
            password (str, optional): The password to use for authentication. Defaults to "dummy".
        """
        super().__init__(command_queue=command_queue, ping_interval=ping_interval)
        self.address = address
        self.port = port
        self.https = https

        # get the token from the server using the username and password. This will be used for authentication.
        self.token = self.request_token(username=username, password=password)

    def start(self):
        pass

    def add_storage_interface(self, storage_interface: FileStorageInterface) -> None:
        self.storage_interface = storage_interface
        self.storage_interface.set_token(self.token)

    def request_token(self, username: str = "", password: str = "") -> str:
        """Request a token from the server.

        Args:
            username (str): The username to use for authentication.
            password (str): The password to use for authentication.

        Returns:
            str: The token.
        """

        try:
            response = requests.post(
                f"{'https' if self.https else 'http'}://{self.address}:{self.port}/token",
                data={"username": username, "password": password},
            )
        except requests.exceptions.ConnectionError as e:
            raise ServerRequestError(f"Could not connect to server")

        if response.status_code == 401:
            raise UnauthorizedError("Invalid username or password")
        elif response.status_code != 200:
            raise RuntimeError(f"Could not get token: {response.text}")
        return response.json()["access_token"]

    def send_status_update(self, status_update: StatusUpdate) -> None:
        try:
            resource_uuids = {}

            if status_update.contains_files():
                resource_uuids.update(
                    self.storage_interface.upload_resources(
                        status_update.response.get_files(),
                        is_server_only=True,
                    )
                )

            response = requests.post(
                f"{'https' if self.https else 'http'}://{self.address}:{self.port}/status",
                json=status_update.unload(resource_uuids).dict(),
                headers={"Authorization": f"Bearer {self.token}"},
            )

        except requests.exceptions.ConnectionError as e:
            raise ServerRequestError(f"Could not connect to server")
        except Exception as e:
            raise ServerRequestError(f"Could not send server request: {e}")

    def send_server_request(self, request: ServerRequest) -> ExecutionResponse:
        """Send a server request to the server.

        Args:
            request (ServerRequest): The server request to send.

        Returns:
            requests.Response: The response from the server.

        Raises:
            ServerRequestError: If the server request could not be sent.
            UnauthorizedError: If the server returns a 401 status code.
        """

        try:
            response = requests.post(
                f"{'https' if self.https else 'http'}://{self.address}:{self.port}/serverrequest",
                json=request.dict(),
                headers={"Authorization": f"Bearer {self.token}"},
            )

            if response.status_code == 401:
                raise UnauthorizedError(response.json()["detail"])

            execution_response = TransmissionExecutionResponse(**response.json())
            files = {}
            if execution_response.contains_files():
                for file_name, file_uuid in execution_response.files.items():
                    response = self.storage_interface.load_resource(file_uuid)
                    files[file_name] = response
            execution_response = ExecutionResponse(
                data=execution_response.data,
                files=files,
                response_type=execution_response.response_type,
            )
            return execution_response

        except requests.exceptions.ConnectionError as e:
            raise ServerRequestError(f"Could not connect to server")
        except Exception as e:
            raise ServerRequestError(f"Could not send server request: {e}")
