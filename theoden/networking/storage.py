from __future__ import annotations

from typing import Annotated
from uuid import UUID, uuid4

import fastapi
import requests
from fastapi import Body, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from ..common import UnauthorizedError
from ..security.auth import AuthenticationManager, UserRole
from ..security.token import create_access_token, decode_token

from .middleware import SecurityHeadersMiddleware

import logging
logger = logging.getLogger(__name__)

class UploadedItem(BaseModel):
    uuid: UUID | str
    name: str
    file: bytes
    is_server_only: bool = False
    username: str | None = None
    permissions: list[str] | None = None


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/storage-token", auto_error=False)


class FileStorage(FastAPI):
    def __init__(
        self,
        persistent: bool = False,
        node_config: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.add_middleware(SecurityHeadersMiddleware)
        self.file_storage: dict[str, UploadedItem] = {}
        self.persistent = persistent

        self.auth_manager = AuthenticationManager(
            yaml_file=node_config, simulation=node_config is None
        )

        @self.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            request: Request, exc: RequestValidationError
        ):
            exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
            logger.error(f"{request}: {exc_str}")
            content = {"status_code": 10422, "message": exc_str, "data": None}
            return JSONResponse(
                # content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
                content=content,
                status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        @self.post("/storage-token")
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

            return {"access_token": access_token, "token_type": "bearer"}

        def authenticate_token(token: str) -> str:
            try:
                username = decode_token(token)
                if not self.auth_manager.get_user_by_name(username):
                    raise UnauthorizedError("Invalid authentication credentials")
            except Exception as e:
                raise HTTPException(
                    status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return username

        @self.post("/file")
        async def file(
            token: Annotated[str, Depends(oauth2_scheme)],
            files: list[UploadFile],
            is_server_only: bool = False,
            users: list[str] = None,
        ) -> dict[str, str]:
            username = authenticate_token(token)

            uuids = {}
            for file in files:
                name = file.filename
                content = await file.read()
                upload_uuid = str(uuid4())
                self.file_storage[upload_uuid] = UploadedItem(
                    uuid=upload_uuid,
                    name=name,
                    file=content,
                    is_server_only=is_server_only,
                    username=username,
                    permissions=users,
                )
                uuids[name] = upload_uuid

            # Return the UUIDs
            return uuids

        @self.get("/file/{upload_uuid}")
        async def file(
            token: Annotated[str, Depends(oauth2_scheme)],
            upload_uuid: UUID,
        ):
            username = authenticate_token(token)

            if (
                # if the file does not exist
                str(upload_uuid) not in self.file_storage
                # if the user is not a server and the file is server-only
                or (
                    self.auth_manager.get_user_by_name(username).role != UserRole.SERVER
                    and self.file_storage[str(upload_uuid)].is_server_only
                )
                # if the user is not the owner of the file and the user is not in the permissions list
                # or (
                #     username
                #     not in (self.file_storage[str(upload_uuid)].permissions or [])
                #     + [self.file_storage[str(upload_uuid)].username]
                #     and not self.auth_manager.simulation
                # )
            ):
                raise HTTPException(
                    status_code=fastapi.status.HTTP_404_NOT_FOUND,
                    detail="File not found",
                )

            response = iter([self.get_file(str(upload_uuid))])

            return StreamingResponse(response, media_type="application/octet-stream")

        @self.delete("/file/{upload_uuid}")
        async def file(
            token: Annotated[str, Depends(oauth2_scheme)],
            upload_uuid: UUID,
        ):
            username = authenticate_token(token)
            if str(upload_uuid) not in self.file_storage or (
                self.auth_manager.get_user_by_name(username).role != UserRole.SERVER
                and self.file_storage[str(upload_uuid)].is_server_only
            ):
                raise HTTPException(
                    status_code=fastapi.status.HTTP_404_NOT_FOUND,
                    detail="File not found",
                )

            del self.file_storage[str(upload_uuid)]
            return Response(status_code=fastapi.status.HTTP_200_OK)

    def get_file(self, upload_uuid: str) -> bytes:
        return self.file_storage[upload_uuid].file


class FileStorageInterface:
    def __init__(
        self,
        storage: FileStorage | None = None,
        address: str = "localhost",
        port: int = 8000,
        https: bool = False,
        username: str = "client",
        password: str = "client",
    ) -> None:
        """Create a new FileStorageInterface.

        Args:
            storage (FileStorage | None, optional): The FileStorage instance to use. Defaults to None.
            address (str, optional): The address of the server. Defaults to "localhost".
            port (int, optional): The port of the server. Defaults to 8000.
            https (bool, optional): Whether to use HTTPS. Defaults to False.
        """

        self.storage = storage
        self.address = address
        self.port = port
        self.https = https

        self.set_token(self.request_token(username, password))

    def set_token(self, token: str) -> None:
        """Set the authentication token.

        Args:
            token (str): The authentication token.
        """

        self.token = token

    def request_token(self, username: str, password: str) -> str:
        """Request an authentication token from the server.

        Args:
            username (str): The username.
            password (str): The password.

        Returns:
            str: The authentication token.
        """

        if self.storage is not None:
            return "dummy"

        response = requests.post(
            f"{'https' if self.https else 'http'}://{self.address}:{self.port}/storage-token",
            data={"username": username, "password": password},
        )

        if response.status_code == 401:
            raise UnauthorizedError("Invalid authentication credentials")

        return response.json()["access_token"]

    def upload_resources(
        self,
        files: dict[str, bytes],
        is_server_only: bool = False,
        permissions: list[str] = None,
    ) -> dict[str, str]:
        """Upload resource_manager to the server.

        Args:
            files (dict[str, bytes]): The files to upload.

        Returns:
            dict[str, str]: A dictionary of file names and their corresponding UUIDs.
        """

        if self.storage is not None:
            upload_uuids = {}
            for file_name, file_content in files.items():
                upload_uuid = str(uuid4())
                self.storage.file_storage[upload_uuid] = UploadedItem(
                    uuid=upload_uuid,
                    name=file_name,
                    file=file_content,
                    is_server_only=is_server_only,
                    permissions=permissions,
                )
                upload_uuids[file_name] = upload_uuid
            return upload_uuids

        if self.token is None:
            raise ValueError("No token provided")

        files = [
            ("files", (file_name, file_content))
            for file_name, file_content in files.items()
        ]

        response = requests.post(
            f"{'https' if self.https else 'http'}://{self.address}:{self.port}/file",
            files=files,
            data={"is_server_only": is_server_only},
            headers={"Authorization": f"Bearer {self.token}"},
        )

        return response.json()

    def load_resource(self, file_uuid: str) -> bytes:
        """Load a resource from the server.

        Args:
            file_uuid (str): The UUID of the file to load.

        Returns:
            bytes: The file content.
        """

        if self.storage is not None:
            return self.storage.get_file(file_uuid)

        if self.token is None:
            raise ValueError("No token provided")

        response = requests.get(
            f"{'https' if self.https else 'http'}://{self.address}:{self.port}/file/{file_uuid}",
            headers={"Authorization": f"Bearer {self.token}"},
        )

        return response.content

    def remove_resource(self, file_uuid: str) -> None:
        """Remove a resource from the server.

        Args:
            file_uuid (str): The UUID of the file to remove.
            token (str): The authentication token.
        """

        if self.storage is not None:
            del self.storage.file_storage[file_uuid]
            return

        if self.token is None:
            raise ValueError("No token provided")

        requests.delete(
            f"{'https' if self.https else 'http'}://{self.address}:{self.port}/file/{file_uuid}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
