from __future__ import annotations

from pydantic import BaseModel


from enum import Enum
from uuid import uuid4
from collections import OrderedDict
import yaml

from ..common import Metadata
from ..common.errors import TopologyError
from ..security import hash_value, verify_hash


class TopologyType(Enum):
    NODE = "node"
    SERVER = "server"


class NodeStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ERROR = "error"
    BLOCKED = "blocked"


class AuthData:
    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        name: str | None = None,
    ) -> None:
        """Class that stores the authentication data of a node.

        Args:
            username (str | None, optional): The username of the node. Defaults to None.
            password (str | None, optional): The password of the node. It will be hashed. Defaults to None.
            name (str | None, optional): The name of the node. Defaults to None

        Raises:
            AssertionError: If the username, password or name are not str or None.
        """
        assert (
            isinstance(username, str) or username is None
        ), f"Username must be a str, or None, got {type(username).__name__}"
        assert (
            isinstance(password, str) or password is None
        ), f"Password must be a str, or None, got {type(password).__name__}"
        assert (
            isinstance(name, str) or name is None
        ), f"Name must be a str, or None, got {type(name).__name__}"

        self.username = username
        self.password_hash = hash_value(password) if password is not None else None
        self.name = name
        self.token: str | None = None

    def authenticate(self, password: str) -> bool:
        return verify_hash(password, self.password_hash)

    def has_token(self) -> bool:
        return self.token is not None

    def has_username_and_password(self) -> bool:
        return self.username is not None and self.password_hash is not None


class NodeData(Metadata):
    def __init__(
        self,
        uuid: str,
        auth: AuthData | None = None,
        data: dict | None = None,
        flags: set[str] | None = None,
        storage: dict | None = None,
        status: NodeStatus = NodeStatus.DISCONNECTED,
        _logging: bool = True,
        **kwargs,
    ):
        super().__init__(
            data={
                **(data if data is not None else {}),
                "flags": flags if flags is not None else set(),
                **kwargs,
            },
            logging=_logging,
        )
        self.uuid = uuid
        self.auth = auth
        if self.auth is None:
            self.auth = AuthData()
        self.status = status
        self._storage = storage if storage is not None else {}

    def has_flag(self, flag: str) -> bool:
        return flag in self["flags"]

    def requires_auth(self) -> bool:
        return self.auth.has_username_and_password()

    @staticmethod
    def get_connected_nodes(nodes: dict[str, NodeData]) -> list[str]:
        return [node for node in nodes if nodes[node].status == NodeStatus.CONNECTED]

    @staticmethod
    def get_by_username(nodes: dict[str, NodeData], username: str) -> NodeData | None:
        _nodes = [
            node for uuid, node in nodes.items() if node.auth.username == username
        ]
        assert len(_nodes) <= 1, f"Found {len(_nodes)} nodes with username {username}"
        return _nodes[0] if len(_nodes) == 1 else None

    def __repr__(self) -> str:
        return f"NodeData(uuid={self.uuid}, has_auth={self.auth is not None}, flags={self['flags']}, status={self.status})"


class TopologySchema(BaseModel):
    auth_mode: bool = False
    allow_new_nodes: bool = True
    max_nodes: int | None = None
    min_nodes: int = 1

    def validate_initial_nodes(self, initial_nodes: list[NodeData]):
        """Validate the initial nodes according to the schema.

        Args:
            initial_nodes (list[NodeData]): The initial nodes.

        Raises:
            TopologyError: If the auth mode is on, but not all nodes have username and password.
            TopologyError: If the min number of nodes is not reached.
            TopologyError: If the max number of nodes is exceeded.
        """
        if self.auth_mode:
            if not all(
                [node.auth.has_username_and_password() for node in initial_nodes]
            ):
                raise TopologyError(
                    "Auth mode is on, but not all nodes have username and password"
                )
        if len(initial_nodes) < self.min_nodes:
            raise TopologyError(
                f"Min number of nodes not reached (min_nodes={self.min_nodes}, got {len(initial_nodes)})"
            )
        if self.max_nodes is not None and len(initial_nodes) > self.max_nodes:
            raise TopologyError(
                f"Max number of nodes exceeded (max_nodes={self.max_nodes}, got {len(initial_nodes)})"
            )

    def validate_new_node(self, number_nodes: int, new_node: NodeData):
        if not self.allow_new_nodes:
            raise TopologyError("New nodes are not allowed")
        if self.max_nodes is not None and number_nodes + 1 >= self.max_nodes:
            raise TopologyError(
                f"Max number of nodes reached (max_nodes={self.max_nodes})"
            )
        if self.auth_mode:
            if not new_node.auth.has_username_and_password():
                raise TopologyError(
                    "Auth mode is on, but the new node does not have username and password"
                )


class TopologyRegister:
    def __init__(
        self,
        initial_nodes: list[AuthData] | str | None = None,
        schema: TopologySchema | None = None,
    ) -> None:
        self._registry = {t: OrderedDict() for t in TopologyType}

        self.schema = schema
        if self.schema is None:
            self.schema = TopologySchema()

        if initial_nodes is not None:
            if isinstance(initial_nodes, str):
                initial_nodes = self.from_yaml(initial_nodes)
            self._register_initial_nodes(initial_nodes)

    def _register_initial_nodes(self, initial_nodes: list[AuthData]) -> None:
        """Register the initial nodes.

        Args:
            initial_nodes (list[AuthData]): The initial nodes.

        Raises:
            TopologyError: If the initial nodes are not valid according to the schema.
        """
        # create NodeData objects
        _initial_nodes = [
            NodeData(
                uuid=str(uuid4()),
                auth=node_auth,
                flags=set(),
                storage={},
            )
            for node_auth in initial_nodes
        ]
        # validate initial nodes according to schema
        self.schema.validate_initial_nodes(_initial_nodes)
        # add to dictionary using the uuid as key
        for node in _initial_nodes:
            self._registry[TopologyType.NODE][node.uuid] = node

    def register(
        self,
        topology_type: TopologyType,
        auth: AuthData | None = None,
        flags: set | None = None,
        storage: dict | None = None,
        **kwargs,
    ) -> NodeData:
        """Register a node or server.

        Args:
            topology_type (TopologyType): The type of the node or server.
            auth (AuthData | None, optional): The authentication data of the node. Defaults to None.
            uuid (str | None, optional): The uuid of the node. Defaults to None.
            flags (set | None, optional): The flags of the node. Defaults to None.
            storage (dict | None, optional): The storage of the node. Defaults to None.

        Returns:
            str | None: The uuid of the node or server.
        """

        if topology_type == TopologyType.NODE:
            new_object = NodeData(
                uuid=str(uuid4()), auth=auth, flags=flags, storage=storage, **kwargs
            )
            self.schema.validate_new_node(len(self.nodes), new_object)
            self.nodes[new_object.uuid] = new_object
        return new_object

    def register_node(
        self,
        auth: AuthData | None = None,
        flags: set | None = None,
        storage: dict | None = None,
        status: NodeStatus = NodeStatus.DISCONNECTED,
        **kwargs,
    ) -> NodeData:
        node = self.register(TopologyType.NODE, auth, flags, storage, **kwargs)
        node.status = status
        return node

    @property
    def nodes(self) -> dict[str, NodeData]:
        return self._registry[TopologyType.NODE]

    @property
    def servers(self) -> dict[str, Metadata]:
        return self._registry[TopologyType.SERVER]

    def set_flag_of_nodes(self, nodes: list[str], flag: str, set_to: bool = True):
        for node in nodes:
            if set_to:
                self.nodes[node]["flags"] = self.nodes[node]["flags"] | {flag}
                self.nodes[node].add_comment("flags", f"Added flag `{flag}`")
            else:
                self.nodes[node]["flags"] = self.nodes[node]["flags"] - {flag}
                self.nodes[node].add_comment("flags", f"Removed flag `{flag}`")

    def get_nodes_with_flag(self, flag: str) -> list[str]:
        return [node for node in self.nodes if flag in self.nodes[node]["flags"]]

    def get_register_of_type(self, type_: TopologyType) -> dict[str, Metadata]:
        return self._registry[type_]

    def get_all_nodes(self, only_uuid: bool = True) -> list[str] | list[str, NodeData]:
        if only_uuid:
            return list(self.nodes.keys())
        else:
            return self.nodes

    def get_connected_nodes(
        self, only_uuid: bool = True, exclude: list[str] | None = None
    ) -> list[str]:
        """Function that returns the uuids of the connected nodes.

        Returns:
            list[str]: The uuids of the connected nodes.
        """
        if exclude is None:
            exclude = []
        nodes = [
            node
            for uuid, node in self.nodes.items()
            if node not in exclude and node.status == NodeStatus.CONNECTED
        ]
        if only_uuid:
            return [node.uuid for node in nodes]
        else:
            return nodes

    def all_nodes_connected(self) -> bool:
        """Function that checks if all nodes are connected.

        Returns:
            bool: True if all nodes are connected.
        """
        return all(
            [node.status == NodeStatus.CONNECTED for node in self.nodes.values()]
        )

    def get_status_of_nodes(self) -> dict[str, NodeStatus]:
        """Returns a dictionary with the status of each node.

        Returns:
            dict[str, NodeStatus]: A dictionary with the status of each node.
        """
        return {uuid: node.status for uuid, node in self.nodes.items()}

    def user_exists(self, uuid: str) -> bool:
        return any(
            [
                node.uuid == uuid if node.auth else False
                for uuid, node in self.nodes.items()
            ]
        )

    def get_node_by_username(self, username: str) -> NodeData:
        """Get a node by its username.

        Args:
            username (str): The username of the node.

        Returns:
            NodeData: The node with the given username.
        """
        return NodeData.get_by_username(self.nodes, username)

    def get_auth_data_by_username(self, username: str) -> AuthData:
        return self.get_node_by_username(username).auth

    def authenticate_user(self, username: str, password: str) -> bool:
        return self.get_auth_data_by_username(username).authenticate(username, password)

    @staticmethod
    def from_yaml(path: str) -> list[AuthData]:
        """Load a list of AuthData objects from a yaml file.

        Args:
            path (str): The path to the yaml file.

        Returns:
            list[AuthData]: A list of AuthData objects.

        Raises:
            AssertionError: If the usernames or names are not unique.
        """
        # load yaml file
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # check that every username  and name is unique
        usernames = [node["username"] for node in data]
        assert len(usernames) == len(set(usernames)), "Usernames are not unique"
        names = [node["name"] for node in data if node["name"] is not None]
        assert len(names) == len(set(names)), "Names are not unique"

        # hash passwords and return AuthData objects
        return [
            AuthData(
                **{
                    "username": node["username"],
                    "password": node["password"],
                    "name": node["name"],
                }
            )
            for node in data
        ]
