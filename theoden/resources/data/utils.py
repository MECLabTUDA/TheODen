import torch


def none_return() -> None:
    return None


def create_pseudo_mask(img_shape):
    return torch.ones((img_shape[1], img_shape[2])).long()
