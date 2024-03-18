from typing import TYPE_CHECKING

from .... import Transferable
from ....common import ExecutionResponse
from ....resources import SampleDataset
from ....resources.data import BalancingDistribution, Partition, PartitionDataset
from ..command import Command

if TYPE_CHECKING:
    from ....operations import DistributionStatusTable


class SetPartitionCommand(Command, Transferable):
    def __init__(
        self,
        partition_function: Partition | None = None,
        balancing_function: BalancingDistribution | None = None,
        partition_key: str | int | None = None,
        key: str = "dataset",
        seed: int = 42,
        use_all_clients: bool = True,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Set the partition on the client. This command will create a new dataset for each partition key.
        Each client will only receive the dataset for its partition key.

        Args:
            partition_function (Partition, optional): The partition function. Defaults to None.
            balancing_function (BalancingDistribution, optional): The balancing function. Defaults to None.
            partition_key (str | int, optional): The partition key. Defaults to None.
            key (str, optional): The key of the dataset. Defaults to "dataset".
            seed (int, optional): The seed. Defaults to 42.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)
        self.partition_function = partition_function
        self.balancing_function = balancing_function
        self.key = key
        self.seed = seed
        self.kwargs = kwargs
        self.partition_key = partition_key
        self.use_all_clients = use_all_clients

    def execute(self) -> ExecutionResponse | None:
        self.client_rm.sr(
            key=self.key,
            resource=PartitionDataset(
                dataset=self.client_rm.gr(self.key, SampleDataset),
                partition_function=self.partition_function,
                balancing_function=self.balancing_function,
                partition_key=self.partition_key,
                seed=self.seed,
                **self.kwargs,
            ).init_after_deserialization(),
        )
        return None

    def client_specific_modification(
        self, distribution_table: "DistributionStatusTable", client_name: str
    ) -> Command:
        included = (
            distribution_table.selected
            if not self.use_all_clients
            else distribution_table.clients
        )
        num_total_clients = len(included)
        partition_key = sorted(included).index(client_name)

        self.balancing_function.add_initialization_parameter(
            number_of_partitions=num_total_clients,
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
        """Set the local partition on the client. This command will create a new local dataset for each partition key.

        Args:
            base_dataset (str, optional): The key of the base dataset to wrap. Defaults to "dataset".
            partition_function (Partition, optional): The partition function. Defaults to None.
            balancing_function (BalancingDistribution, optional): The balancing function. Defaults to None.
            seed (int, optional): The seed. Defaults to 42.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)
        self.base_dataset = base_dataset
        self.partition_function = partition_function
        self.balancing_function = balancing_function
        self.seed = seed

    def execute(self) -> ExecutionResponse | None:
        # copy base dataset
        base = self.client_rm.gr(self.base_dataset)

        for key in self.balancing_function.keys():
            self.client_rm.sr(
                key=f"{self.base_dataset}:{key}",
                resource=PartitionDataset(
                    dataset=base,
                    partition_key=key,
                    partition_function=self.partition_function,
                    balancing_function=self.balancing_function,
                    seed=self.seed,
                ).init_after_deserialization(),
            )
