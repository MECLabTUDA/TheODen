import torch

from typing import List
import json
import inspect
from hashlib import sha224

IMAGENET_MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
IMAGENET_STD = [x / 255 for x in [63.0, 62.1, 66.7]]


def none_return() -> None:
    """Returns None."""
    return None


def create_sorted_lists(
    dict1: dict[str, any], dict2: dict[str, any]
) -> tuple[list, list]:
    """Returns two lists of values from two dictionaries, sorted by key.

    Args:
        dict1 (dict[str, any]): The first dictionary.
        dict2 (dict[str, any]): The second dictionary.

    Returns:
        tuple[list, list]: The two lists of values. Sorted by key.
    """

    sorted_keys = sorted(set(dict1.keys()))
    sorted_dict1 = [dict1.get(key) for key in sorted_keys]
    sorted_dict2 = [dict2.get(key) for key in sorted_keys]
    return sorted_dict1, sorted_dict2


def hash_dict(d: dict[str, any], method: callable = sha224) -> str:
    """
    Returns a hash of a dictionary.

    Args:
        d (dict): The dictionary to hash.
        method (callable, optional): The hash function to use. Defaults to sha256.

    Returns:
        str: The hash of the dictionary.
    """
    return method(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def last_index(lst: List[str], string: str):
    """
    Returns the index of the last occurrence of a given string in a list.

    Parameters
    ----------
    lst : list
        The list to search in.
    string : str
        The string to search for.

    Returns
    -------
    int
        The index of the last occurrence of the string in the list, or -1 if the string is not found.

    Examples
    --------
    >>> lst = ['apple', 'banana', 'orange', 'banana', 'kiwi']
    >>> string = 'banana'
    >>> last_index(lst, string)
    3
    """
    try:
        return len(lst) - 1 - lst[::-1].index(string)
    except ValueError:
        return -1


def are_classes_from_same_package(class1: type, class2: type) -> bool:
    """Returns whether two classes are from the same package.

    Args:
        class1 (type): The first class.
        class2 (type): The second class.

    Returns:
        bool: Whether the two classes are from the same package.
    """

    module1 = inspect.getmodule(class1)
    module2 = inspect.getmodule(class2)
    return module1.__name__.split(".")[0] == module2.__name__.split(".")[0]


def to_list(obj) -> list:
    """Converts an object to a list.

    Args:
        obj (any): The object to convert to a list.

    Returns:
        list: The object converted to a list.
    """
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def calculate_min_max_mean_of_state_dict(
    state_dict: dict[str, any]
) -> tuple[float, float, float]:
    """Calculates the minimum, maximum and mean of a state dict.

    Args:
        state_dict (dict[str, any]): The state dict.

    Returns:
        tuple[float, float, float]: The minimum, maximum and mean of the state dict.
    """
    vals = []
    for key in state_dict:
        vals.append(state_dict[key].flatten())
    test = torch.cat(vals)
    return torch.min(test).item(), torch.max(test).item(), torch.mean(test).item()
