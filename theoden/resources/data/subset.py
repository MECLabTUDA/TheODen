import numpy as np
from tqdm import tqdm
from typing_extensions import Self

from .dataset import SampleDataset


class SubsetDataset(SampleDataset):
    def __init__(
        self,
        dataset: SampleDataset,
        indices: list[int] | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.requested = False

    def get_indices(
        self,
        folder: str,
        mask_fn: callable,
        show_progress: bool = True,
        description: str = "",
        invert: bool = False,
        force: bool = False,
    ) -> list[int]:
        if not force:
            indices = self.load_indices_from_fingerprint(folder)
        else:
            indices = None
        if indices is None:
            index_mask = self.apply_fn_to_all_samples(
                mask_fn,
                show_progress=show_progress,
                description=description,
            )
            indices = np.arange(len(self.dataset))[
                index_mask if not invert else [not elem for elem in index_mask]
            ]
            self.save_fingerprint(folder, {"indices": indices.tolist()})
        return indices

    def apply_fn_to_all_samples(
        self, fn: callable, show_progress: bool = True, description: str = ""
    ) -> list[bool]:
        indices_values: list[bool] = []
        for sample in (
            tqdm(self.dataset, desc=description) if show_progress else self.dataset
        ):
            indices_values.append(fn(sample))
        return indices_values

    def load_indices_from_fingerprint(self, folder: str) -> list[int] | None:
        try:
            fingerprint = self.load_fingerprint(folder)
        except FileNotFoundError:
            return None
        return fingerprint["indices"]

    def init_on_data_request(self) -> None:
        pass

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)
