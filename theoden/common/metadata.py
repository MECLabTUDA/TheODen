import pandas as pd

from collections import defaultdict
from typing import Any, List, Dict, Optional, Union
from datetime import datetime
from enum import Enum
from copy import deepcopy

from .utils import none_return


class ChangeType(Enum):
    CREATED = 1
    CHANGED = 2
    REMOVED = 3
    REVERT = 4


class Metadata(dict):
    def __init__(self, data: dict | None = None, logging: bool = False):
        super().__init__()
        self.logging = logging
        self.log = (
            pd.DataFrame(columns=["time", "key", "value", "comment", "change_type"])
            if logging
            else None
        )
        if data:
            for key, value in data.items():
                self[key] = value

    @staticmethod
    def _savable(value: any) -> bool:
        # check if it useful to store in dataframe or too big, e.g. don't store tensors but int and bools
        if isinstance(value, (int, float, bool, str, list, dict, type(None))):
            return True
        return False

    def _append_log(self, key: str, value: Any, change_type: ChangeType) -> None:
        self.log = pd.concat(
            [
                self.log,
                pd.DataFrame(
                    [
                        {
                            "time": datetime.now(),
                            "key": key,
                            "value": deepcopy(value)
                            if self._savable(value)
                            else type(value).__name__,
                            "comment": None,
                            "change_type": change_type,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Overrides the default __setitem__ method to log the changes made to the keys
        #"""
        # if self.logging:
        #     if key not in self.log["key"].values:
        #         change_type = ChangeType.CREATED
        #     else:
        #         change_type = ChangeType.CHANGED
        #     self._append_log(key, value, change_type)
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """
        Overrides the default __delitem__ method to log the changes made to the keys
        """
        if self.logging:
            self._append_log(key, None, ChangeType.REMOVED)
        super().__delitem__(key)

    def add_comment(self, key: str, comment: str) -> None:
        """Add a comment to the last change made to the key

        Args:
            key (str): The key to add the comment to
            comment (str): The comment to add
        """
        # if self.logging:
        #     last_row = self.log[self.log["key"] == key].iloc[-1]
        #     last_row["comment"] = comment
        #     self.log.iloc[-1] = last_row

    def revert_changes(self, key: str, n: int = 1) -> None:
        """Revert the last n changes made to the key or if key is None revert the last change made to any key

        Args:
            key (str): The key to revert the changes of
            n (int, optional): The number of changes to revert. Defaults to 1.

        Raises:
            ValueError: If n is larger than the number of changes made to the key
        """
        if self.logging:
            df_key = self.log[self.log["key"] == key]
            if not df_key.empty:
                last_n = df_key.tail(n + 1)
                if n <= len(last_n) - 1:
                    self[key] = last_n.iloc[-n - 1]["value"]
                    self.log.drop(self.log.tail(1).index, inplace=True)
                    self._append_log(key, self[key], ChangeType.REVERT)
                    self.add_comment(key, f"Revert {n} change(s)")

    def get_log(self, key: str | None = None) -> List[pd.DataFrame]:
        """Returns the log of changes made to the keys in a DataFrame format

        Args:
            key (str, optional): The key to get the log of. Defaults to None.

        Returns:
            List[pd.DataFrame]: A list of DataFrames containing the log of changes made to the keys
        """
        if key:
            df_key = self.log[self.log["key"] == key]
            if not df_key.empty:
                return [df_key]
            else:
                return []
        else:
            keys = self.log["key"].unique()
            return [self.log[self.log["key"] == k] for k in keys]

    def __repr__(self) -> str:
        return f"Metadata({super().__repr__()})"
