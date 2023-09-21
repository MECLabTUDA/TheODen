import torch

# from .sample import Sample


def none_return() -> None:
    return None


def create_pseudo_mask(img_shape):
    return torch.ones((img_shape[1], img_shape[2])).long()


# def full_image_ood(sample: Sample, keep_ignorred):
#     if keep_ignorred and sample.metadata["ignore_index"] != None:
#         ignore_map = sample["ood_mask"] == sample.metadata["ignore_index"]
#     sample["ood_mask"] = torch.zeros_like(sample["ood_mask"])
#     if keep_ignorred and sample.metadata["ignore_index"] != None:
#         sample["ood_mask"][ignore_map] = sample.metadata["ignore_index"]
#     return sample
