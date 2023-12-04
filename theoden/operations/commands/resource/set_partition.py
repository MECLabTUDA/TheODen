from ....common import ExecutionResponse
from ....resources import SampleDataset
from .... import Transferable
from ....resources.data import (
    PartitionDataset,
    Partition,
    PartitionDataset,
    BalancingDistribution,
)
from ..command import Command


class SetPartitionCommand(Command, Transferable):
    def __init__(
        self,
        partition_function: Partition | None = None,
        balancing_function: BalancingDistribution | None = None,
        partition_key: str | int | None = None,
        key: str = "dataset",
        seed: int = 42,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(uuid=uuid, **kwargs)
        self.partition_function = partition_function
        self.balancing_function = balancing_function
        self.key = key
        self.seed = seed
        self.kwargs = kwargs
        self.partition_key = partition_key

    def execute(self) -> ExecutionResponse | None:
        self.node_rm.sr(
            key=self.key,
            resource=PartitionDataset(
                dataset=self.node_rm.gr(self.key, SampleDataset),
                partition_function=self.partition_function,
                balancing_function=self.balancing_function,
                partition_key=self.partition_key,
                seed=self.seed,
                **self.kwargs,
            ).init_after_deserialization(),
        )
        return None

    def node_specific_modification(
        self, distribution_table: "DistributionStatusTable", node_name: str
    ) -> Command:
        included = distribution_table.selected
        num_total_nodes = len(included)
        partition_key = sorted(included).index(node_name)

        self.balancing_function.add_initialization_parameter(
            number_of_partitions=num_total_nodes,
            _overwrite=True,
        )
        self.add_initialization_parameter(
            partition_key=partition_key,
            _overwrite=True,
        )
        return self


class SetLocalPartitionCommand(Command, Transferable):
    def __init__(
        self,
        base_dataset: str = "dataset",
        partition_function: Partition | None = None,
        balancing_function: BalancingDistribution | None = None,
        seed: int = 42,
        *,
        uuid: str | None = None,
        **kwargs,
    ):
        super().__init__(uuid=uuid, **kwargs)
        self.base_dataset = base_dataset
        self.partition_function = partition_function
        self.balancing_function = balancing_function
        self.seed = seed

    def execute(self) -> ExecutionResponse | None:
        # copy base dataset
        base = self.node_rm.gr(self.base_dataset)

        for key in self.balancing_function.keys():
            self.node_rm.sr(
                key=f"{self.base_dataset}:{key}",
                resource=PartitionDataset(
                    dataset=base,
                    partition_key=key,
                    partition_function=self.partition_function,
                    balancing_function=self.balancing_function,
                    seed=self.seed,
                ).init_after_deserialization(),
            )
