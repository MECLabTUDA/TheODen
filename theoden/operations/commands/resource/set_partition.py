from typing import Optional, Any, Type
from theoden.common import ExecutionResponse

from theoden.resources import SampleDataset
from theoden.resources.data import BalancingDistribution, Partition, PercentageBalancing


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
        seed: Optional[int] = 42,
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            node=node,
            uuid=uuid,
            **kwargs,
        )
        self.partition_function = partition_function
        self.balancing_function = balancing_function
        self.key = key
        self.seed = seed
        self.kwargs = kwargs
        self.partition_key = partition_key

    def execute(self) -> Any:
        self.node_rr.sr(
            key=self.key,
            resource=PartitionDataset(
                dataset=self.node_rr.gr(self.key, SampleDataset),
                partition_function=self.partition_function,
                balancing_function=self.balancing_function,
                partition_key=self.partition_key,
                seed=self.seed,
                **self.kwargs,
            ).init_after_deserialization(),
        )

    def node_specific_modification(
        self, status_register: dict[str, "StatusTable"], node_uuid: str
    ) -> Command:
        table = status_register[self.uuid]
        included = table.get_included()
        num_total_nodes = len(included)
        partition_key = sorted(included).index(node_uuid)

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
        seed: Optional[int] = 42,
        *,
        node: Optional["Node"] = None,
        uuid: str | None = None,
        **kwargs,
    ):
        super().__init__(node=node, uuid=uuid, **kwargs)
        self.base_dataset = base_dataset
        self.partition_function = partition_function
        self.balancing_function = balancing_function
        self.seed = seed

    def execute(self) -> ExecutionResponse | None:
        # copy base dataset
        base = self.node_rr.gr(self.base_dataset)

        for key in self.balancing_function.keys():
            self.node_rr.sr(
                key=f"{self.base_dataset}:{key}",
                resource=PartitionDataset(
                    dataset=base,
                    partition_key=key,
                    partition_function=self.partition_function,
                    balancing_function=self.balancing_function,
                    seed=self.seed,
                ).init_after_deserialization(),
            )


class SetLocalSplitCommand(SetLocalPartitionCommand, Transferable):
    def __init__(
        self,
        base_dataset: str,
        partition_function: Partition | None = None,
        seed: int | None = 42,
        *,
        node: Any | None = None,
        uuid: str | None = None,
        **kwargs,
    ):
        super().__init__(
            base_dataset,
            partition_function,
            PercentageBalancing(kwargs),
            seed,
            node=node,
            uuid=uuid,
            **kwargs,
        )
