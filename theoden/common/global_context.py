import yaml
import multiprocessing

from .singleton import SingletonMeta


class GlobalContext(metaclass=SingletonMeta):
    def __init__(self):
        self._manager = multiprocessing.Manager()
        self._context: dict[str, any] = self._manager.dict()

    def set(self, key: str, value: any):
        self._context[key] = value

    def get(self, key: str, default: any = ...) -> any:
        if key not in self._context:
            if default is ...:
                raise KeyError(f"Key `{key}` not found in global context")
            else:
                return default
        return self._context[key]

    def __getitem__(self, key: str) -> any:
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def load_from_yaml(self, path: str):
        with open(path, "r") as f:
            self._context = yaml.load(f, Loader=yaml.FullLoader)

    def get_dataset_path(self, dataset: str, parameter_path: str | None = None) -> str:
        """Get the path to a dataset.

        Args:
            dataset (str): The name of the dataset.
            parameter_path (str | None, optional): The path to the dataset. Defaults to None.

        Returns:
            str: The path to the dataset.

        Raises:
            KeyError: If the dataset is not found in the global context and no default dataset path is set.
        """

        if parameter_path is not None:
            return parameter_path

        if "datasets" not in self._context:
            raise KeyError("No datasets found in global context")
        # check if dataset is in global context
        if dataset not in self._context["datasets"]:
            if "DEFAULT" in self._context["datasets"]:
                return self._context["datasets"]["DEFAULT"]
            raise KeyError(
                f"Dataset `{dataset}` not found in global context and no default dataset path"
            )
        return self._context["datasets"][dataset]
