from ....common import Transferable
from ....resources.data import DataSampler
from .set_resource import SetResourceCommand


class SetDataSamplerCommand(SetResourceCommand, Transferable):
    def __init__(
        self,
        datasampler: DataSampler,
        dataset: str = "dataset:train_sampler",
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Set the data sampler on the client

        Args:
            datasampler (DataSampler): The data sampler to set
            dataset (str, optional): The resource key of the data sampler. Defaults to "dataset:train_sampler".
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(
            key=dataset,
            resource=datasampler,
            overwrite=True,
            unpack_dict=False,
            uuid=uuid,
            **kwargs,
        )

        if not isinstance(self.resource, DataSampler):
            raise TypeError(
                f"Resource must be of type DataSampler, not {type(self.resource)}"
            )
