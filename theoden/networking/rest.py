from __future__ import annotations

import fastapi
from fastapi import FastAPI, Request, status, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette.datastructures import UploadFile as StarletteUploadFile

import logging
import requests
from typing import TYPE_CHECKING, Annotated

from ..operations import *
from ..common import (
    Transferables,
    StatusUpdate,
    RegisteredTypeModel,
    UnauthorizedError,
    ServerRequestError,
)
from .interface import NodeInterface
from ..topology.topology_register import TopologyRegister, NodeData, NodeStatus
from ..security import create_access_token, decode_token


if TYPE_CHECKING:
    from ..operations import ServerRequest
    from ..topology.server import Server


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)


class RestServerInterface(FastAPI):
    def __init__(self, server: "Server", **kwargs) -> None:
        super().__init__(**kwargs)
        self.server = server

        @self.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            request: Request, exc: RequestValidationError
        ):
            exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
            logging.error(f"{request}: {exc_str}")
            content = {"status_code": 10422, "message": exc_str, "data": None}
            return JSONResponse(
                # content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
                content=content,
                status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        @self.post("/token")
        async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
            if not self.top_reg.schema.auth_mode:
                # We dont need to authenticate, so we ignore the username and password
                node_data = self.top_reg.register_node(status=NodeStatus.CONNECTED)
                # create a new access token that does not expire since we are in local mode
                access_token = create_access_token(
                    data={"sub": node_data.uuid}, delta=0
                )

            else:
                # We need to authenticate, so we check the username and password
                node_data = self.top_reg.get_node_by_username(form_data.username)
                if not node_data:
                    raise HTTPException(
                        status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                        detail="Incorrect username or password",
                    )
                if not node_data.auth.authenticate(form_data.password):
                    raise HTTPException(
                        status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                        detail="Incorrect username or password",
                    )
                # create a new access token
                access_token = create_access_token(data={"sub": node_data.uuid})

                node_data.token = access_token
                node_data.status = NodeStatus.CONNECTED

            return {"access_token": access_token, "token_type": "bearer"}

        def authenticate(token: str) -> str:
            node_uuid = decode_token(token)
            if not self.top_reg.user_exists(node_uuid):
                raise HTTPException(
                    status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return node_uuid

        @self.post("/status")
        async def status_with_files(
            token: Annotated[str, Depends(oauth2_scheme)], request: Request
        ):
            # authenticate the user
            node_uuid = authenticate(token)

            # get the form data
            form_data = await request.form()

            # convert the form data to a dict
            obj_dict = {
                key: value
                for key, value in form_data.items()
                if not isinstance(value, StarletteUploadFile)
            }
            obj_dict["node_uuid"] = node_uuid

            try:
                # convert the response to a dict
                obj_dict["response"] = {}
                obj_dict["response"]["data"] = form_data["response"]

                # convert the files to bytes
                obj_dict["response"]["files"] = {
                    key: await value.read()
                    for key, value in form_data.items()
                    if isinstance(value, StarletteUploadFile)
                }
                obj_dict["response"]["response_type"] = form_data.get(
                    "response_type", None
                )
                # convert the dict to a StatusUpdateWithFile
                status_update = StatusUpdate.parse_obj(obj_dict)

            except Exception as e:
                print(e)
                raise HTTPException(
                    status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid status update",
                )

            return self.server.process_status_update(status_update)

        @self.post("/serverrequest")
        async def server_request(
            token: Annotated[str, Depends(oauth2_scheme)],
            request_body: RegisteredTypeModel,
        ):
            # authenticate the user
            node_uuid = authenticate(token)

            # convert the request body to a ServerRequest
            server_request = Transferables().to_object(
                request_body.dict(),
                "ServerRequest",
                server=self.server,
                node_uuid=node_uuid,
            )
            try:
                # process the server request on the server
                response = self.server.process_server_request(server_request)
            except UnauthorizedError as e:
                raise HTTPException(
                    status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                    detail=str(e),
                )

            # if response are bytes, return a file response, otherwise return a json response
            if isinstance(response, bytes):
                return StreamingResponse(iter([response]))
            elif isinstance(response, str):
                return FileResponse(response)
            else:
                return JSONResponse(response)

    @property
    def top_reg(self) -> TopologyRegister:
        return self.server.topology_register


class RestNodeInterface(NodeInterface):
    def __init__(
        self,
        address: str = "localhost",
        port: int = 8000,
        https: bool = False,
        username: str = "dummy",
        password: str = "dummy",
    ) -> None:
        """A federated learning node communication interface based on REST.

        Args:
            address (str, optional): The address of the server. Defaults to "localhost".
            port (int, optional): The port of the server. Defaults to 8000.
            https (bool, optional): Whether to use https. Defaults to False.
            username (str, optional): The username to use for authentication. Defaults to "dummy".
            password (str, optional): The password to use for authentication. Defaults to "dummy".
        """

        self.address = address
        self.port = port
        self.https = https

        # get the token from the server using the username and password. This will be used for authentication.
        self.token = self.request_token(username=username, password=password)

    def send_status_update(self, status_update: StatusUpdate) -> requests.Response:
        try:
            response = requests.post(
                f"{'https' if self.https else 'http'}://{self.address}:{self.port}/status",
                **status_update.to_post(),
                headers={"Authorization": f"Bearer {self.token}"},
            )
        except requests.exceptions.ConnectionError as e:
            raise ServerRequestError(f"Could not connect to server")
        except Exception as e:
            raise ServerRequestError(f"Could not send server request: {e}")
        # if response.status_code == 401:
        #     # check if token expired
        #     if response.json()["detail"] == "Invalid authentication credentials":
        #     self.request_token()
        #     self.send_status_update(status_update)

    def send_server_request(self, request: "ServerRequest") -> requests.Response:
        try:
            response = requests.post(
                f"{'https' if self.https else 'http'}://{self.address}:{self.port}/serverrequest",
                json=request.dict(),
                headers={"Authorization": f"Bearer {self.token}"},
            )
        except requests.exceptions.ConnectionError as e:
            raise ServerRequestError(f"Could not connect to server")
        except Exception as e:
            raise ServerRequestError(f"Could not send server request: {e}")

        if response.status_code == 401:
            raise UnauthorizedError(response.json()["detail"])
        return response

    def request_token(self, username: str = "", password: str = "") -> str:
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
