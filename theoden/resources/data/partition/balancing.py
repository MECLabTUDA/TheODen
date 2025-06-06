import random

import numpy as np

from ....common import Transferable


class BalancingDistribution(Transferable, is_base_type=True):
    def keys(self, **kwargs) -> list[str]:
        raise NotImplementedError("Please implement this method")

    def __call__(
        self, partition_indices: dict[str, list[int]], seed: int | None = None, **kwargs
    ) -> dict[str, list[str]]:
        raise NotImplementedError("Please implement this method")


class PercentageBalancing(BalancingDistribution):
    def __init__(self, percentage: dict[str | int, float]) -> None:
        self.percentage = percentage

    def keys(self, **kwargs) -> list[str | int]:
        return list(self.percentage.keys())

    def __call__(
        self, partition_indices: dict[str, list[int]], seed: int | None = None, **kwargs
    ) -> dict[str, list[str]]:
        # shuffle the indices
        keys = list(partition_indices.keys())
        random.seed(seed)
        random.shuffle(keys)
        random.seed(None)

        # calculate the size based on the percentage
        sizes = {}
        for key, percentage in self.percentage.items():
            sizes[key] = int(len(keys) * percentage)

        # calculate the remainder
        remainder = len(keys) - sum(sizes.values())

        # add the remainder to the first partition
        sizes[next(iter(self.percentage))] += remainder

        # create the indices
        indices = {}
        for key, size in sizes.items():
            indices[key] = keys[:size]
            keys = keys[size:]

        return indices


class EqualBalancing(BalancingDistribution):
    def __init__(
        self, number_of_partitions: int | None = None, split_along_key: bool = True
    ) -> None:
        self.number_of_partitions = number_of_partitions
        self.split_along_key = split_along_key

    def keys(self, **kwargs) -> list[int]:
        return list(range(self.number_of_partitions or kwargs.get("num_partitions")))

    def __call__(
        self, partition_indices: dict[str, list[int]], seed: int | None = None, **kwargs
    ) -> dict[str, list[str]]:
        # split equally
        if self.split_along_key:
            # get the number of partitions from the arguments or the class
            num = (
                self.number_of_partitions
                if self.number_of_partitions is not None
                else kwargs.get("num_partitions", None)
            )
            if num is None:
                raise ValueError("The number of partitions must be specified")

            # split along the key
            keys = list(partition_indices.keys())
            random.seed(seed)
            random.shuffle(keys)
            random.seed(None)
            # calculate the fraction size and the remainder
            fraction_size = len(keys) // num
            # the remainder is the number of partitions that will have an extra sample
            remainder = len(keys) % num
            # create a list of the size of each partition
            parts = [fraction_size] * num
            # add the extra samples to the partitions
            for i in range(remainder):
                parts[i] += 1
            split_indices = {}
            start = 0
            for i, part in enumerate(parts):
                split_indices[i] = keys[start : start + part]
                start += part
            return split_indices
        else:
            raise NotImplementedError("Not implemented yet")


class DirichletBalancing(BalancingDistribution):
    def __init__(self, alpha: dict[str | int, float]) -> None:
        self.alpha = alpha

    def __call__(
        self, partition_indices: dict[str, list[int]], seed: int | None = None, **kwargs
    ) -> dict[str, list[str]]:
        return super().__call__(partition_indices, seed, **kwargs)


class DiscreteBalancing(BalancingDistribution):
    def __init__(self, balancing: dict[str | int, list[str]]) -> None:
        self.balancing = balancing

    def keys(self, **kwargs) -> list[str | int]:
        return list(self.balancing.keys())

    def __call__(
        self, partition_indices: dict[str, list[int]], seed: int | None = None, **kwargs
    ) -> dict[str, list[str]]:
        return self.balancing


class KeyBalancing(BalancingDistribution):
    def keys(self, **kwargs) -> list[str | int]:
        raise ValueError("The keys are only known after the partitioning")

    def __call__(
        self, partition_indices: dict[str, list[int]], seed: int | None = None, **kwargs
    ) -> dict[str, list[str]]:
        return {self.key: partition_indices[self.key]}


class LogNormalBalancing(BalancingDistribution):
    def __init__(self, sigma: float, number_of_partitions: int | None = None) -> None:
        self.sigma = sigma
        self.number_of_partitions = number_of_partitions

    def keys(self, **kwargs) -> list[str | int]:
        return list(range(self.number_of_partitions or kwargs.get("num_partitions")))

    def __call__(
        self, partition_indices: dict[str, list[int]], seed: int | None = None, **kwargs
    ) -> dict[str, list[str]]:
        num_samples_per_client = int(len(partition_indices.keys()) / len(self.keys()))

        client_sample_nums = np.random.lognormal(
            mean=np.log(num_samples_per_client), sigma=self.sigma, size=len(self.keys())
        )
        client_sample_nums = (
            client_sample_nums
            / np.sum(client_sample_nums)
            * len(partition_indices.keys())
        ).astype(int)
        diff = np.sum(client_sample_nums) - len(partition_indices.keys())  # diff <= 0

        # Add/Subtract the excess number starting from first client
        if diff != 0:
            for cid in range(len(self.keys())):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break
