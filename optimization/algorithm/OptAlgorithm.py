"""
Date: 2024-04-09 11:00:39
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-11 16:15:37
"""

from typing import Any, Generator
import matplotlib.pyplot as plt
import pandas as pd
import tqdm as tqdm
from abc import abstractmethod, ABC

from ..solution import Solution
from ..problem import Problem


class OptAlgorithm(ABC):
    def __init__(
        self,
        problem: Problem,
        pop_size: int = 100,
    ) -> None:
        """优化算法抽象基类

        Parameters
        ----------
        problem : Problem
            问题类, 包含目标函数和约束函数等信息
        pop_size : int, optional
            种群大小, by default 100
        """
        self._problem: Problem = problem
        self._pop_size: int = pop_size

        self._population: list[Solution] = []
        self._bests: list[Solution] = []
        self._FEs_list: list[int] = []

    @property
    def population(self) -> list[Solution]:
        return self._population

    @property
    def bests(self) -> list[Solution]:
        return self._bests

    @property
    def FEs_list(self) -> list[int]:
        return self._FEs_list

    @abstractmethod
    def _search(self) -> list[Solution]:
        """子类算法必须实现基于当前种群对新解的搜索

        Returns
        -------
        list[Solution]
            返回新解的列表
        """
        ...

    def _select(self, new_pop: list[Solution]) -> None:
        """对当前种群和新种群的选择, 选择后的种群直接保存再当前种群, 子类算法可选重写

        Parameters
        ----------
        new_pop : list[Solution]
            新种群
        """
        for i in range(self._pop_size):
            if new_pop[i].obj < self._population[i].obj:
                self._population[i] = new_pop[i]

    def _evaluate_pop(self, population: list[Solution] | None = None) -> None:
        """对种群中的每个解进行目标函数和约束函数的计算

        Parameters
        ----------
        population : list[Solution] | None, optional
            待评估的种群, 当输入None时评估当前种群, by default None
        """
        if population is None:
            population = self._population

        for solution in population:
            self._problem.evaluate(sol=solution)

    def _to_dataframe(self, population: list[Solution] | None = None) -> pd.DataFrame:
        """将种群转化成DataFrame格式, 便于输出结果

        Returns
        -------
        pd.DataFrame
            population的DataFrame格式
        """
        pop_to_csv: list[Solution] = (
            self._population if population is None else population
        )

        pop: Generator[tuple[float, float, *tuple[Any, ...]], None, None] = (
            (solution.obj, solution.con) + tuple(x for x in solution.dec)
            for solution in pop_to_csv
        )

        columns: list[str] = ["obj", "con"] + [
            f"x{i}" for i in range(self._problem.dim)
        ]

        return pd.DataFrame(data=pop, columns=columns)

    def solve(self, max_FEs: int = int(1e6)) -> None:
        """算法运行, 子类算法可选重写

        Parameters
        ----------
        max_FEs : int, optional
            算法运行最大的评估次数(总共搜索解的数量), by default int(1e6)
        """
        pbar = tqdm.tqdm(total=max_FEs, unit="FEs")

        self._bests = []
        self._FEs_list = []
        FEs = 0
        self._population: list[Solution] = self._problem.init_population(self._pop_size)

        self._evaluate_pop()

        FEs += self._pop_size
        self._bests.append(min(self._population, key=lambda x: x.obj))
        self._FEs_list.append(FEs)

        pbar.set_description(f"Best solution: {self._bests[-1].obj:.2e}")
        pbar.update(self._pop_size)

        while FEs < max_FEs:
            new_pop: list[Solution] = self._search()
            self._evaluate_pop(population=new_pop)
            self._select(new_pop)
            FEs += self._pop_size
            pbar.set_description(desc=f"Best solution: {self._bests[-1].obj:.2e}")
            pbar.update(self._pop_size)
            self._bests.append(min(self._population, key=lambda x: x.obj))
            self._FEs_list.append(FEs)

        print()
        print(f"Best solution: {self._bests[-1]}")

    def draw(self) -> None:
        """对求解的结果进行绘图, 子类算法可选重写"""
        plt.plot(self._FEs_list, [x.obj for x in self._bests])
        plt.xlabel(xlabel="FEs")
        plt.ylabel(ylabel="Objective Value")
        plt.title(label="Optimization Process")
        plt.show()

    def output(self, filename: str = "result/result.csv") -> None:
        """将当前种群存在文件中

        Parameters
        ----------
        filename : str, optional
            待存文件路径, by default result/result.csv"
        """
        df: pd.DataFrame = self._to_dataframe()
        df.sort_values(by=["obj", "con"], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(path_or_buf=filename)
        print(f"Solution saved in {filename}")
