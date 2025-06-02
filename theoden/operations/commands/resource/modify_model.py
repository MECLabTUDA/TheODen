from ....resources import TorchModel, WrappedTorchModel
from .. import Command

import logging
logger = logging.getLogger(__name__)


class SelectTorchEncoderOfModelCommand(Command):

    def __init__(
        self,
        base_model_key: str = "model",
        encoder_key: str | None = None,
        overwrite: bool = True,
        *,
        uuid: str | None = None,
        **kwargs
    ) -> None:
        super().__init__(uuid=uuid, **kwargs)
        self.base_model_key = base_model_key
        self.encoder_key = encoder_key
        self.overwrite = overwrite

    def execute(self) -> None:

        base_model = self.client_rm.gr(
            self.base_model_key, assert_type=TorchModel
        ).module()

        try:
            encoder_model = TorchModel().set_model(base_model.encoder)
        except Exception as e:
            logger.error(
                "Error when loading the encoder. Ensure the base model ha a property encoder."
            )
            raise e

        self.client_rm.sr(self.encoder_key or self.base_model_key, encoder_model)
        return None


class WrapModelCommand(Command):
    def __init__(
        self,
        wrapped_model_key: str,
        wrapper_class: type[WrappedTorchModel],
        model_key: str = "model",
        *,
        uuid: str | None = None,
        **model_kwargs
    ) -> None:
        super().__init__(uuid=uuid, **model_kwargs)
        self.wrapped_model_key = wrapped_model_key
        self.model_key = model_key
        self.model_kwargs = model_kwargs
        self.wrapper_class = wrapper_class

    def execute(self) -> None:
        self.client_rm.sr(
            self.model_key,
            self.wrapper_class(
                wrapped_model=self.client_rm.gr(self.wrapped_model_key, TorchModel),
                **self.model_kwargs
            ).parse_to(self.client_rm.gr("device", str)),
        )
        return None
