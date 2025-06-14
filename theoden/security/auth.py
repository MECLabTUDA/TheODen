import urllib.parse
from enum import Enum
from uuid import uuid4

import requests
import yaml

from ..common import UnauthorizedError
from .hash import hash_value, verify_hash

import logging
logger = logging.getLogger(__name__)

class UserRole(Enum):
    SERVER = "server"
    CLIENT = "client"
    OBSERVER = "observer"


class User:
    def __init__(
        self, username: str, password: str, role: UserRole, token: str | None = None
    ):
        self.username = username
        self.password = password
        self.role = role
        self.token = token

    def __repr__(self):
        return f"User(username={self.username}, role={self.role})"


class AuthenticationManager:
    def __init__(self, yaml_file: str = None, simulation: bool = False):
        # Initialize an empty dictionary to store user information.
        self.users: dict[str, User] = {}
        self.simulation = simulation

        # Load user data from a YAML file if provided
        if yaml_file is not None:
            self.load_users_from_yaml(yaml_file)

        # If auth_all_if_empty is True, then all users will be authenticated if the user dictionary is empty.
        # This is useful for testing and simulation purposes but should be set to False in production.
        if self.simulation:
            logger.warning(
                "AuthenticationManager is in simulation mode. All users will be authenticated and created if they do not exist."
            )

            # add dummy server user
            self.register_user("server", "server", UserRole.SERVER)

    def register_user(
        self, username: str, password: str, role: UserRole, is_hashed: bool = False
    ) -> User:
        """
        Register a new user with a username, password, and role.

        Args:
            username (str): The username of the user.
            password (str): The plaintext password for the user.
            role (UserRole): The role of the user (UserRole.SERVER, UserRole.CLIENT, or UserRole.OBSERVER).
        """
        if username in self.users:
            raise ValueError("Username already exists. Please choose a different one.")
        else:
            self.users[username] = User(
                username=username,
                password=hash_value(password) if not is_hashed else password,
                role=role,
            )
            return self.users[username]

    def authenticate(self, username: str, password: str, role: UserRole = None) -> User:
        """
        Authenticate a user based on their username, password, and optionally, role.

        Args:
            username (str): The username of the user.
            password (str): The plaintext password for the user.
            role (UserRole, optional): The required role for authentication.

        Returns:
            bool: True if authentication is successful, False otherwise.
        """

        user = self.get_user_by_name(username)

        if self.simulation:
            return user

        if user:
            if verify_hash(password, user.password):
                if role is None or user.role == role.value:
                    logger.info(
                        f"Authentication successful for user '{username}' with role '{user.role}'."
                    )
                    return user
                else:
                    raise UnauthorizedError(
                        f"Authentication failed. User '{username}' does not have the required role '{role}'."
                    )
            else:
                raise UnauthorizedError("Authentication failed. Incorrect password.")
        else:
            raise UnauthorizedError("Authentication failed. User not found.")

    def get_user_by_name(self, username: str) -> User | None:
        # If simulation is True, then all users are authenticated and created if they do not exist.
        if self.simulation and username not in self.users:
            return self.register_user(str(uuid4()), "dummy", UserRole.CLIENT)

        # return user if exists else return None
        return self.users.get(username, None)

    def get_user_role(self, username: str) -> UserRole:
        """
        Get the role of a user.

        Args:
            username (str): The username of the user.

        Returns:
            UserRole: The role of the user.
        """
        if username in self.users:
            return UserRole(self.users[username].role)
        else:
            raise ValueError("User not found.")

    def load_users_from_yaml(self, yaml_file: str):
        """
        Load user data from a YAML file and populate the users dictionary.

        Args:
            yaml_file (str): The path to the YAML file containing user data.
        """
        with open(yaml_file, "r") as file:
            users_data = yaml.safe_load(file)

        if users_data is not None:
            for user in users_data:
                name = user["name"]
                password = user["password"]
                role = user["role"]
                self.register_user(name, password, UserRole(role), is_hashed=True)

    @staticmethod
    def create_yaml_and_users(
        yaml_file: str,
        output_file: str,
        create_users: bool = False,
        api_url: str = "http://localhost:15672/api",
        api_user: str = "guest",
        api_password: str = "guest",
        vhost: str = "",
    ):
        with open(yaml_file, "r") as file:
            users = yaml.safe_load(file)

        if create_users:
            vhost_url = f"{api_url}/api/vhosts/{vhost}"
            response = requests.delete(vhost_url, auth=(api_user, api_password))
            response = requests.put(vhost_url, auth=(api_user, api_password))

        yaml_users = []
        for user in users:
            name = user["name"]
            password = user["password"]
            role = user["role"]

            yaml_users.append(
                {"name": name, "role": role, "password": hash_value(password)}
            )

            if create_users:
                AuthenticationManager.create_rabbitmq_user(
                    name,
                    password,
                    UserRole(role),
                    api_url=api_url,
                    api_user=api_user,
                    api_password=api_password,
                    vhost=vhost,
                )

        with open(output_file, "w") as file:
            yaml.dump(yaml_users, file)

    @staticmethod
    def create_rabbitmq_user(
        username: str,
        password: str,
        role: UserRole,
        api_url: str = "http://localhost:15672/api",
        api_user: str = "guest",
        api_password: str = "guest",
        vhost: str = "/",
    ):
        """
        Create a RabbitMQ user and set permissions based on the user's role.

        Args:
            username (str): The username of the user.
            password (str): The plaintext password for the user.
            role (UserRole): The role of the user (UserRole.SERVER, UserRole.CLIENT, or UserRole.OBSERVER).
            api_url (str, optional): The URL of the RabbitMQ Management API. Defaults to "http://localhost:15672/api".
            api_user (str, optional): The username of the RabbitMQ Management API user. Defaults to "guest".
            api_password (str, optional): The password of the RabbitMQ Management API user. Defaults to "guest".
            vhost (str, optional): The name of the virtual host. Defaults to "theoden".
        """

        # Define user tags based on the role
        if role == UserRole.SERVER:
            user_tags = "administrator"
        else:
            user_tags = ""

        # Create the user using RabbitMQ Management API
        user_data = {"password": password, "tags": user_tags}
        user_url = f"{api_url}/users/{username}"
        response = requests.put(
            user_url,
            auth=(api_user, api_password),
            headers={"Content-Type": "application/json"},
            json=user_data,
        )

        if response.status_code == 201:
            logger.info(f"RabbitMQ user '{username}' created successfully.")
        elif response.status_code == 204:
            logger.info(f"RabbitMQ user '{username}' already existed; password/tags updated.")
        else:
            logger.error(
                f"Failed to create RabbitMQ user '{username}'. Status code: {response.status_code}"
            )
        regex_pattern = f"{username}.*"
        # Set permissions for the user
        permissions_data = (
            {"configure": ".*", "write": ".*", "read": ".*"}
            if role == UserRole.SERVER
            else {
                "configure": regex_pattern,
                "write": regex_pattern,
                "read": regex_pattern,
            }
        )

        permissions_url = f"{api_url}/permissions/%2F{urllib.parse.quote(vhost)}/{urllib.parse.quote(username)}"

        response = requests.put(
            permissions_url,
            auth=(api_user, api_password),
            headers={"Content-Type": "application/json"},
            json=permissions_data,
        )

        if response.status_code == 201:
            logger.info(f"RabbitMQ permissions for '{username}' set successfully.")
        elif response.status_code == 204:
            logger.info(f"RabbitMQ user '{username}' already existed; password/tags updated.")
        else:
            logger.error(
                f"Failed to set RabbitMQ permissions for '{username}'. Status code: {response.status_code}"
            )
