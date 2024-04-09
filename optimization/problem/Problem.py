"""
Date: 2024-04-08 15:17:56
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-09 21:55:23
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
        """问题类

        Parameters
        ----------
        func : Callable[[np.ndarray], float]
            目标函数
        bounds : np.ndarray
            边界, 2-D array
        con_func : Callable[[np.ndarray], float] | None, optional
            约束函数, None代表只在边界条件内, by default None
        repair_func : Callable[[np.ndarray], np.ndarray] | None, optional
            修复无效解函数, None代表只修复到边界内, by default None

        Raises
        ------
        DimensionError
            代表输入的边界不是2-D array
        """
        self.__func: Callable[[np.ndarray], float] = func
        self.__bounds: np.ndarray = bounds
        self.__con_func: Callable[[np.ndarray], float] | None = con_func
        self.__repair_func: Callable[[np.ndarray], np.ndarray] | None = repair_func

        if bounds.shape[0] != 2:
            raise DimensionError("bounds must be a 2-D array")
        self.__dim: int = bounds.shape[1]

    def __cal_obj(self, x: np.ndarray) -> float:
        """计算目标函数

        Parameters
        ----------
        x : np.ndarray
            决策向量

        Returns
        -------
        float
            目标值
        """
        return self.__func(x)

    def __cal_con(self, x: np.ndarray) -> float:
        """计算约束值

        Parameters
        ----------
        x : np.ndarray
            决策向量

        Returns
        -------
        float
            约束值
        """
        return (
            self.__con_func(x) if self.__con_func is not None else self.__default_con(x)
        )

    def __default_con(self, x: np.ndarray) -> float:
        """默认约束函数

        Parameters
        ----------
        x : np.ndarray
            决策向量

        Returns
        -------
        float
            约束值
        """
        return float(any(x > self.__bounds[1, :]) or any(x < self.__bounds[0, :]))

    def __default_repair(self, x: np.ndarray) -> np.ndarray:
        """默认修复函数

        Parameters
        ----------
        x : np.ndarray
            决策向量

        Returns
        -------
        np.ndarray
            修复后的决策向量
        """
        return np.clip(x, a_min=self.__bounds[0, :], a_max=self.__bounds[1, :])

    def __repair(self, x: np.ndarray) -> np.ndarray:
        """修复函数

        Parameters
        ----------
        x : np.ndarray
            决策向量

        Returns
        -------
        np.ndarray
            修复后决策向量
        """
        return (
            self.__repair_func(x)
            if self.__repair_func is not None
            else self.__default_repair(x)
        )

    @property
    def dim(self) -> int:
        return self.__dim

    def init_population(self, size: int) -> list[Solution]:
        """初始化种群, 提供给算法类的接口

        Parameters
        ----------
        size : int
            种群大小

        Returns
        -------
        list[Solution]
            初始化后的种群
        """
        return [
            Solution(dec=x)
            for x in np.random.uniform(
                low=self.__bounds[0, :],
                high=self.__bounds[1, :],
                size=(size, self.__dim),
            )
        ]

    def evaluate(self, sol: Solution) -> None:
        """评估一个解, 提供给算法类的接口, 结果保存在当前解

        Parameters
        ----------
        sol : Solution
            待评估的解
        """
        sol.dec = sol.dec if self.__cal_con(sol.dec) <= 0 else self.__repair(sol.dec)
        sol.obj = self.__cal_obj(sol.dec)
        sol.con = self.__cal_con(sol.dec)
