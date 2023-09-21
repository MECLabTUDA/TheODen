from typing import Optional

from ....common import ExecutionResponse, Transferable
from ....resources.data import DataSampler
from .set_resource import SetResourceCommand


class SetDataSamplerCommand(SetResourceCommand, Transferable):
    def __init__(
        self,
        datasampler: DataSampler,
        dataset: str = "dataset:train_sampler",
        *,
        node: Optional["Node"] = None,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            key=dataset,
            resource=datasampler,
            overwrite=True,
            unpack_dict=False,
            node=node,
            uuid=uuid,
            **kwargs,
        )

        if not isinstance(self.resource, DataSampler):
            raise TypeError(
                f"Resource must be of type DataSampler, not {type(self.resource)}"
            )
