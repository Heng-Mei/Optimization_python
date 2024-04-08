"""
Date: 2024-04-08 15:17:56
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-08 17:10:04
"""

from typing import Callable
import numpy as np

from ..solution.Solution import Solution


class Problem:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        con_func: Callable[[np.ndarray], float] | None = None,
        repair_func: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.__func: Callable[[np.ndarray], float] = func
        self.__bounds: np.ndarray = bounds
        self.__con_func: Callable[[np.ndarray], float] | None = con_func
        self.__repair_func: Callable[[np.ndarray], np.ndarray] | None = repair_func

    def __cal_obj(self, x: np.ndarray) -> float:
        return self.__func(x)

    def __cal_con(self, x: np.ndarray) -> float:
        return (
            self.__con_func(x) if self.__con_func is not None else self.__default_con(x)
        )

    def __default_con(self, x: np.ndarray) -> float:
        return float(any(x > self.__bounds[1, :]) or any(x < self.__bounds[0, :]))

    def __default_repair(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.__bounds[0, :], self.__bounds[1, :])

    def __repair(self, x: np.ndarray) -> np.ndarray:
        return (
            self.__repair_func(x)
            if self.__repair_func is not None
            else self.__default_repair(x)
        )

    @property
    def dim(self) -> int:
        return self.__bounds.shape[1]

    def init_population(self, size: int) -> np.ndarray:
        return np.random.uniform(
            self.__bounds[0, :], self.__bounds[1, :], (size, self.__bounds.shape[1])
        )

    def evaluate(self, x: np.ndarray) -> Solution:
        x = x if self.__cal_con(x) <= 0 else self.__repair(x)
        obj: float = self.__cal_obj(x)
        con: float = self.__cal_con(x)
        return Solution(x, obj, con)
