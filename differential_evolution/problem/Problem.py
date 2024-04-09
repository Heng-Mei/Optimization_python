"""
Date: 2024-04-08 15:17:56
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-09 16:40:51
"""

from typing import Callable
import numpy as np

from ..solution.Solution import Solution
from .._exception.problem_exceptions import DimensionError


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

        if bounds.shape[0] != 2:
            raise DimensionError("bounds must be a 2-D array")
        self.__dim: int = bounds.shape[1]

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
        return self.__dim

    def init_population(self, size: int) -> list[Solution]:
        return [
            Solution(x)
            for x in np.random.uniform(
                self.__bounds[0, :], self.__bounds[1, :], (size, self.__dim)
            )
        ]

    def evaluate(self, sol: Solution) -> None:
        sol.dec = sol.dec if self.__cal_con(sol.dec) <= 0 else self.__repair(sol.dec)
        sol.obj = self.__cal_obj(sol.dec)
        sol.con = self.__cal_con(sol.dec)
