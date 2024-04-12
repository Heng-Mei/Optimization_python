"""
Date: 2024-04-08 15:17:45
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-10 11:48:19
"""

import random as random
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
import tqdm as tqdm

from ..solution import Solution
from ..problem import Problem
from .OptAlgorithm import OptAlgorithm


class DE(OptAlgorithm):
    def __init__(
        self,
        problem: Problem,
        pop_size: int = 100,
        mutation: float = 0.7,
        crossover: float = 0.9,
    ) -> None:
        """继承OptAlgorithm的Differential Evolution算法

        Parameters
        ----------
        problem : Problem
            问题类, 包含目标函数和约束函数等信息
        pop_size : int, optional
            种群大小, by default 100
        mutation : float, optional
            变异概率, by default 0.7
        crossover : float, optional
            交叉, by default 0.9
        """
        super().__init__(problem=problem, pop_size=pop_size)
        self.__mutation: float = mutation
        self.__crossover: float = crossover

    def __mutate(self) -> list[Solution]:
        """DE算法变异

        Returns
        -------
        list[Solution]
            由当前种群变异出的新种群
        """
        new_pop: list[Solution] = []
        for i in range(self._pop_size):
            temp: list[Solution] = random.sample(population=self._population, k=3)
            new_pop.append(
                Solution(temp[0].dec + self.__mutation * (temp[1].dec - temp[2].dec))
            )
        return new_pop

    def __cross(self, new_pop: list[Solution]) -> None:
        """DE算法的交叉

        Parameters
        ----------
        new_pop : list[Solution]
            待与当前种群交叉的新种群, 交叉后的解直接保存在new_pop中
        """
        for i in range(self._pop_size):
            if random.random() > self.__crossover:
                new_pop[i].dec = self._population[i].dec

    def _search(self) -> list[Solution]:
        """重写父类的搜索方法

        Returns
        -------
        list[Solution]
            通过变异和交叉得到的新种群
        """
        new_pop: list[Solution] = self.__mutate()
        self.__cross(new_pop)
        self._evaluate_pop(population=new_pop)
        return new_pop

    # def draw(self) -> None:
    #     """重写父类的绘制方法, 主要修改了绘图的标题"""
    #     plt.plot(
    #         self._FEs_list, [x.obj for x in self._bests], marker="o", ms=4, label="Best"
    #     )
    #     plt.xlabel(xlabel="FEs")
    #     plt.ylabel(ylabel="Objective Value")
    #     plt.title(label="Differential Evolution")
    #     plt.legend()
    #     plt.show()
