from __future__ import annotations

import matplotlib.pyplot as plt

from ...common import Metadata


class MetadataBatch:
    def __init__(self, batch: list[Metadata]) -> None:
        self.batch = batch
        # only the created keys are saved here
        self.created: list[str] = []

    def __len__(self) -> int:
        return len(self.batch)

    def __getitem__(self, arg: str) -> list[any]:
        """Get the value of a key for all metadata in the batch

        Args:
            arg (str): The key to get the value for

        Returns:
            list[any]: The values of the key for all metadata in the batch

        Example:
            >>> metadata = MetadataBatch([Metadata({"key": "value"}), Metadata({"key": "value2"})])
            >>> metadata["key"]
            ["value", "value2"]
        """
        return [m[arg] for m in self.batch]

    def __setitem__(self, idx: str, value: any) -> None:
        """Set the same value to the same key in all metadata

        Args:
            idx (str): The key to set
            value (any): The value to set

        Example:
            >>> metadata = MetadataBatch([Metadata(), Metadata()])
            >>> metadata["key"] = "value"
            >>> metadata
            MetadataBatch([{"key": "value"}, {"key": "value"}])
        """
        for m in self.batch:
            m[idx] = value

    def __delitem__(self, idx: str) -> None:
        """Delete a key in all metadata

        Args:
            idx (str): The key to delete

        Example:
            >>> metadata = MetadataBatch([Metadata({"key": "value"}), Metadata({"key": "value2"})])
            >>> metadata["key"]
            ["value", "value2"]
            >>> del metadata["key"]
            >>> metadata["key"]
            ValueError: "key" is not in metadata
        """
        for m in self.batch:
            del m[idx]

    def __contains__(self, idx: str) -> bool:
        """Check if a key is in all metadata

        Args:
            idx (str): The key to check

        Returns:
            bool: Whether the key is in all metadata
        """
        return all(idx in m for m in self.batch)

    def add_comment(self, key: str, comment: str) -> MetadataBatch:
        """Add a comment to a key in the metadata

        Args:
            key (str): The key to add the comment to
            comment (str): The comment to add

        Returns:
            MetadataBatch: The metadata batch with the comment added
        """
        for m in self.batch:
            m.add_comment(key, comment)

        return self

    def init_as_dict(self, key: str) -> MetadataBatch:
        """Initialize a key in the metadata as a dictionary

        Args:
            key (str): The key to initialize

        Returns:
            MetadataBatch: The metadata batch with the key initialized
        """
        for m in self.batch:
            if key not in m:
                m[key] = {}

        return self

    def init_as_list(self, key: str) -> MetadataBatch:
        """Initialize a key in the metadata as a list

        Args:
            key (str): The key to initialize

        Returns:
            MetadataBatch: The metadata batch with the key initialized
        """
        for m in self.batch:
            if key not in m:
                m[key] = []

        return self

    def set_elementwise(self, key: str, values: list[any]) -> MetadataBatch:
        """Set a value to a key in the metadata

        Args:
            key (str): The key to set
            values (list): The values to set

        Example:
            >>> metadata = MetadataBatch([Metadata(), Metadata()])
            >>> metadata.set_elementwise("key", [1, 2])
            >>> metadata
            MetadataBatch([Metadata({'key': 1}), Metadata({'key': 2})])
            >>> metadata.set_elementwise("key", [3, 4])
            >>> metadata
            MetadataBatch([Metadata({'key': 3}), Metadata({'key': 4})])

        Returns:
            MetadataBatch: The metadata batch with the key set
        """

        assert len(self) == len(values), "Length of values must match length of batch"

        # for each metadata in the batch, set the value to the key
        for i, m in enumerate(self.batch):
            m[key] = values[i]

        return self

    def append_to_keyed_dict(
        self, key: str, dict_key: str, items: list
    ) -> MetadataBatch:
        """Insert a value to a key in a dict in the metadata

        Args:
            key (str): The key of the dict
            dict_key (str): The key in the dict
            items (list): The items to insert

        Example:
            >>> metadata = MetadataBatch([Metadata(), Metadata()])
            >>> metadata.append_to_keyed_dict("key", "dict_key", [1, 2])
            >>> metadata
            MetadataBatch([Metadata({'key': {'dict_key': 1}}), Metadata({'key': {'dict_key': 2}})])
            >>> metadata.append_to_keyed_dict("key", "dict_key", [3, 4])
            >>> metadata
            MetadataBatch([Metadata({'key': {'dict_key': 3}}), Metadata({'key': {'dict_key': 4}})])
            >>> metadata.append_to_keyed_dict("key", "dict_key2", [5, 6])
            >>> metadata
            MetadataBatch([Metadata({'key': {'dict_key': 3, 'dict_key2': 5}}), Metadata({'key': {'dict_key': 4, 'dict_key2': 6}})])

        Returns:
            MetadataBatch: The metadata batch with the items inserted
        """

        assert len(self) == len(items), "Length of items must match length of batch"

        # create the dict if it doesn't exist
        if key not in self.created:
            self.init_as_dict(key)

        # add the items to the dict
        for i, m in enumerate(self.batch):
            m[key][dict_key] = items[i]

        return self

    def plot_distribution(
        self,
        key: str,
        bins: int | list[int] = 10,
        figsize: tuple[float, float] = (8, 6),
        **kwargs,
    ) -> plt.Figure:
        """Plot a histogram of the data

        Args:
            key (str): The key to plot
            bins (int | list[int], optional): The number of bins or the bins to use. Defaults to 10.
            figsize (tuple[float, float], optional): The size of the figure. Defaults to (8, 6).

        Returns:
            plt.Figure: The figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(self[key], bins=bins, **kwargs)
        ax.set_xlabel(key)
        ax.set_ylabel("Count")
        return fig

    def __repr__(self) -> str:
        return f"MetadataBatch({self.batch})"


def init_sample_metadata(
    metadata: Metadata | MetadataBatch | None = None,
) -> Metadata | MetadataBatch:
    if not metadata:
        return Metadata()
    else:
        assert isinstance(
            metadata, Metadata | MetadataBatch
        ), "Metadata needs to be class Metadata or MetadataBatch"
        return metadata
