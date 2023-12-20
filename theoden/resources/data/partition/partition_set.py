import itertools

from ....common import GlobalContext, Transferable
from ..dataset import SampleDataset
from ..subset import SubsetDataset
from .balancing import BalancingDistribution
from .partitions import IndexPartition, Partition


class PartitionDataset(SubsetDataset, Transferable, build=False):
    def __init__(
        self,
        dataset: SampleDataset,
        partition_key: str | int,
        partition_function: Partition | None = None,
        balancing_function: BalancingDistribution | None = None,
        seed: int = 42,
        force_overwrite: bool = False,
        **kwargs,
    ):
        # check the arguments
        assert isinstance(dataset, SampleDataset), "Dataset must be a SampleDataset"

        self.remove_initialization_parameter(*kwargs.keys(), "force_overwrite")

        self.partition_function = partition_function
        self.balancing_function = balancing_function
        self.partition_key = partition_key
        self.force_overwrite = force_overwrite

        self.name = self.initialization_hash()

        if self.partition_function is None:
            self.partition_function = IndexPartition()
        else:
            assert isinstance(
                self.partition_function, Partition
            ), "Partition must be a Partition"

        self.seed = seed

        # Get base dataset. Based on this dataset, we will create and store the partitions
        base_dataset = dataset.get_dataset_chain(True)

        # Get the indices for the partitions using the partition function
        indices = self.partition_function(
            dataset=base_dataset, force_overwrite=self.force_overwrite, **kwargs
        )

        # print(indices.keys())

        # partition the indices using the distribution function
        partitions = self.balancing_function(
            partition_indices=indices, seed=self.seed, **kwargs
        )

        if len(partitions[self.partition_key]) < 50:
            print(
                f"Partition [{partition_key}] with partitions {partitions[self.partition_key]}"
            )

        # use partition key to get own partition
        indices = self._partition_to_indices(
            indices=indices, partition=partitions[self.partition_key]
        )

        self.save_fingerprint(
            folder=GlobalContext()["partition_folder"],
            additional_fields={"indices": indices},
        )

        # call super with the indices
        super().__init__(dataset, indices)

    def _partition_to_indices(
        self, indices: dict[str, list[int]], partition: list[int | str]
    ) -> list[int]:
        return list(itertools.chain.from_iterable([indices[i] for i in partition]))
