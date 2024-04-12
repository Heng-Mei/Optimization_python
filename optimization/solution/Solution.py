"""
Date: 2024-04-09 22:19:31
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-12 16:27:08
"""

"""
Date: 2024-04-08 15:18:04
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-12 15:51:56
"""

from typing import Iterable
import numpy as np

from .._exception.solution_exceptions import *

_ObjectLike = float | Iterable | None


class Solution:
    def __init__(
        self,
        dec: np.ndarray,
        obj: _ObjectLike = None,
        con: float | None = None,
    ) -> None:
        """解的类

        Parameters
        ----------
        dec : np.ndarray
            决策向量
        obj : float | None, optional
            目标值, 未被评估为None, by default None
        con : float | None, optional
            约束值, 未被评估为None, by default None
        """
        self.__dec = dec
        self.__obj: _ObjectLike = obj
        self.__con: float | None = con

    def __str__(self) -> str:
        return (
            f"Decision: {self.__dec}, Objective: {self.__obj}, Constraint: {self.__con}"
        )

    def __lt__(self, other: "Solution") -> bool:
        """比较重载, 若obj为Iterable, 则比较所有元素, 否则比较单个元素

        Parameters
        ----------
        other : Solution
            待比较解

        Returns
        -------
        bool
            True if self < other

        Raises
        ------
        ObjectError
            If 'obj' is not all Iterable
        """
        assert self.__obj is not None, "Objective not set"
        assert other.__obj is not None, "Objective not set"

        if isinstance(self.__obj, Iterable) and isinstance(other.__obj, Iterable):
            return all(np.array(self.__obj) < np.array(other.__obj))
        elif isinstance(self.__obj, Iterable) or isinstance(other.__obj, Iterable):
            raise ObjectError("'obj' is not all Iterable")
        else:
            return self.__obj < other.__obj

    @property
    def dec(self) -> np.ndarray:
        return self.__dec

    @dec.setter
    def dec(self, value: np.ndarray) -> None:
        self.__dec = value

    @property
    def obj(self) -> _ObjectLike:
        assert self.__obj is not None, "Objective not set"
        return self.__obj

    @obj.setter
    def obj(self, value: _ObjectLike) -> None:
        self.__obj = value

    @property
    def con(self) -> float:
        assert self.__con is not None, "Constraint not set"
        return self.__con

    @con.setter
    def con(self, value: float) -> None:
        self.__con = value
