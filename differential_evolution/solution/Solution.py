"""
Date: 2024-04-08 15:18:04
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-08 15:38:22
"""

import numpy as np


class Solution:
    def __init__(self, dec: np.ndarray, obj: float, con: float) -> None:
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

    @property
    def obj(self) -> float:
        return self.__obj

    @property
    def con(self) -> float:
        return self.__con
