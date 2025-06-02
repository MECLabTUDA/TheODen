from __future__ import annotations

import time
from enum import Enum
from threading import Thread
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import yaml

from ..resources import ResourceManager
from ..watcher import TopologyChangeNotification
from .client_status import ClientStatusObserver

if TYPE_CHECKING:
    from ..operations import Distribution
    from ..watcher import WatcherPool

import logging
logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"


class NodeType(Enum):
    SERVER = "server"
    CLIENT = "client"


class Node:
    def __init__(
        self,
        node_name: str,
        node_type: NodeType,
        flags: list[str] | None = None,
        data: dict | None = None,
        status: NodeStatus = NodeStatus.OFFLINE,
    ) -> None:
        self.name = node_name
        self.type = node_type
        self.flags = flags or []
        self.data = data or {}
        self.status = status
        self.last_active = time.time()

    @property
    def is_online(self) -> bool:
        """Returns True if the node is online, False otherwise"""
        return self.status == NodeStatus.ONLINE


class Topology:
    def __init__(
        self,
        watcher_pool: "WatcherPool",
        resource_manager: ResourceManager,
        node_config: str | None = None,
        observer: ClientStatusObserver | None = None,
        **kwargs,
    ) -> None:
        self.nodes: dict[str, Node] = (
            Topology.load_from_yaml(yaml_file=node_config) or {}
        )
        self.lifecycle_pool: set["Distribution"] = set()
        self.watcher = watcher_pool
        self.resource_manager = resource_manager

        self.observer = (
            Thread(target=observer.observe, args=(self,), daemon=True)
            if observer
            else None
        )
        self.observer.start() if self.observer else None

    @staticmethod
    def load_from_yaml(yaml_file: str | None) -> dict[str, Node] | None:
        if yaml_file is None:
            return None

        logger.info(f"Loading topology from {yaml_file}")

        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)

        nodes = {}
        for node in yaml_data:
            nodes[node["name"]] = Node(
                node_name=node["name"], node_type=NodeType(node["role"])
            )
        return nodes

    def add_lifecycle(self, lifecycle: "Distribution") -> Topology:
        self.lifecycle_pool.add(lifecycle)
        return self

    def remove_lifecycle(self, lifecycle: "Distribution") -> Topology:
        if lifecycle in self.lifecycle_pool:
            self.lifecycle_pool.remove(lifecycle)
        return self

    def _inform_about_change(self, node_name: str, include_pool: bool = True):
        # update lifecycle pool
        if include_pool:
            for lifecycle in self.lifecycle_pool.copy():
                lifecycle.handle_topology_change(
                    node_name, topology=self, resource_manager=self.resource_manager
                )
        self.watcher.notify_all(TopologyChangeNotification(topology=self))

    def add_node(self, node: Node) -> Topology:
        if node.name in self.nodes:
            raise ValueError(f"Node with name {node.name} already exists")
        self.nodes[node.name] = node
        self._inform_about_change(node.name)
        return self

    def remove_node(self, node_name: str) -> None:
        if node_name not in self.nodes:
            return

        del self.nodes[node_name]
        self._inform_about_change(node_name)

    @property
    def server(self) -> Node:
        for node in self.nodes.values():
            if node.type == NodeType.SERVER:
                return node
        raise ValueError("No server node found")

    @property
    def clients(self) -> list[Node]:
        return [node for node in self.nodes.values() if node.type == NodeType.CLIENT]

    @property
    def client_names(self) -> list[str]:
        return [node.name for node in self.clients]

    @property
    def num_clients(self) -> int:
        return len(self.clients)

    @property
    def num_connected_clients(self) -> int:
        return len([node for node in self.clients if node.status == NodeStatus.ONLINE])

    @property
    def num_offline_clients(self) -> int:
        return len([node for node in self.clients if node.status == NodeStatus.OFFLINE])

    @property
    def fraction_connected_clients(self) -> float:
        return self.num_connected_clients / self.num_clients

    def online_clients(self, names: bool = False) -> list[Node] | list[str]:
        return [
            (node.name if names else node)
            for node in self.clients
            if node.status == NodeStatus.ONLINE
        ]

    def get_client_by_name(self, node_name: str) -> Node:
        for node in self.clients:
            if node.name == node_name:
                return node
        raise KeyError(f"Client with name {node_name} not found")

    def get_clients_with_flag(self, flag: str) -> list[Node]:
        return [node for node in self.clients if flag in node.flags]

    def get_clients_without_flag(self, flag: str) -> list[Node]:
        return [node for node in self.clients if flag not in node.flags]

    def get_clients_with_status(self, status: NodeStatus) -> list[Node]:
        return [node for node in self.clients if node.status == status]

    def flags_of_clients(self) -> dict[str, list[str]]:
        return {client.name: client.flags for client in self.clients}

    def get_flags_of_client(self, node_name: str | Node) -> list[str]:
        if isinstance(node_name, str):
            node_name = self.get_client_by_name(node_name)
        return node_name.flags

    def set_online(self, node_name: str | Node) -> None:
        if isinstance(node_name, str):
            node_name = self.get_client_by_name(node_name)
        node_name.last_active = time.time()
        node_name.status = NodeStatus.ONLINE

        logger.info(f"Node {node_name.name} is online")

        self._inform_about_change(node_name.name)

    def set_offline(self, node_name: str | Node) -> None:
        if isinstance(node_name, str):
            node_name = self.get_client_by_name(node_name)
        node_name.status = NodeStatus.OFFLINE

        logger.info(f"Node {node_name.name} is offline")

        self._inform_about_change(node_name.name)

    def set_flag_of_nodes(self, nodes: list[Node] | list[str], flag: str) -> None:
        for node in nodes:
            _node = node if isinstance(node, Node) else self.get_client_by_name(node)
            if flag not in _node.flags:
                _node.flags.append(flag)
            self._inform_about_change(_node.name, include_pool=False)

    def remove_flag_of_nodes(self, nodes: list[Node] | list[str], flag: str) -> None:
        for node in nodes:
            _node = node if isinstance(node, Node) else self.get_client_by_name(node)
            if flag not in _node.flags:
                _node.flags.remove(flag)
            self._inform_about_change(_node.name, include_pool=False)

    @staticmethod
    def get_nodes_with_flag(nodes: list[Node], flag: str) -> list[Node]:
        return [node for node in nodes if flag in node.flags]

    @staticmethod
    def get_data_of_nodes(nodes: list[Node], data_key: str) -> list[any]:
        return [node.data[data_key] for node in nodes]

    def plot_topology(self) -> plt.Figure:
        G = nx.Graph()

        # Add server node
        server_node = self.server
        G.add_node(
            server_node.name,
            label=server_node.name,
            type=server_node.type.name,
            status=server_node.status.name,
            flags="",
        )

        # Add client nodes and edges
        online_clients = []
        offline_clients = []

        for client in self.clients:
            G.add_node(
                client.name,
                label=client.name,
                type=client.type.name,
                status=client.status.name,
                flags=", ".join(client.flags),
            )
            if client.status == NodeStatus.ONLINE:
                G.add_edge(server_node.name, client.name)
                online_clients.append(client.name)
            else:
                offline_clients.append(client.name)

        # Create a mapping of node labels
        labels = {
            node: f"{node}\nType: {G.nodes[node]['type']}\nStatus: {G.nodes[node]['status']}\nFlags: {G.nodes[node]['flags']}"
            for node in G.nodes
        }

        # Use Kamada-Kawai layout for improved positioning with scale adjustment
        pos = nx.kamada_kawai_layout(G, scale=0.2)  # Adjust the scale for less padding

        # Increase figure size
        plt.figure(figsize=(7, 5))  # Increase the figure size to accommodate more nodes
        pad = 0.3
        plt.xlim(-pad, pad)  # Adjust the x-axis limits to add less padding
        plt.ylim(-pad, pad)  # Adjust the y-axis limits to add less padding

        # Draw the graph with customized labels and node colors
        node_colors = [
            (
                "red"
                if node == server_node.name
                else "green" if G.nodes[node]["status"] == "ONLINE" else "gray"
            )
            for node in G.nodes
        ]
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=250)

        # Customize label appearance
        node_labels = nx.draw_networkx_labels(
            G, pos, labels, font_size=8, verticalalignment="bottom"
        )

        # Add a box behind labels
        for _, label in node_labels.items():
            label.set_bbox(
                {
                    "boxstyle": "round,pad=0.3",
                    "edgecolor": "black",
                    "facecolor": "white",
                    "alpha": 0.75,
                }
            )

        # Draw edges between the server and online clients with a different color
        edge_colors = [
            (
                "yellow"
                if edge[0] == server_node.name and edge[1] in online_clients
                else "gray"
            )
            for edge in G.edges()
        ]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[edge for edge in G.edges() if edge[0] == server_node.name],
            edge_color=edge_colors,
        )
        figure = plt.gcf()
        plt.close(figure)
        return figure
