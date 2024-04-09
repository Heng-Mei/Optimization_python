"""
Date: 2024-04-08 15:18:04
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-09 16:37:36
"""

import numpy as np


class Solution:
    def __init__(
        self, dec: np.ndarray, obj: float | None = None, con: float | None = None
    ) -> None:
        self.__dec = dec
        self.__obj = obj
        self.__con = con

    def __str__(self) -> str:
        return (
            f"Decision: {self.__dec}, Objective: {self.__obj}, Constraint: {self.__con}"
        )

    @property
    def dec(self) -> np.ndarray:
        return self.__dec

    @dec.setter
    def dec(self, value: np.ndarray) -> None:
        self.__dec = value

    @property
    def obj(self) -> float:
        assert self.__obj is not None, "Objective not set"
        return self.__obj

    @obj.setter
    def obj(self, value: float) -> None:
        self.__obj = value

    @property
    def con(self) -> float:
        assert self.__con is not None, "Constraint not set"
        return self.__con

    @con.setter
    def con(self, value: float) -> None:
        self.__con = value
