"""
Date: 2024-04-08 15:17:45
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-08 17:29:44
"""

import numpy as np
import random as random
import matplotlib.pyplot as plt
import tqdm as tqdm

from differential_evolution.solution import Solution
from ..problem import Problem


class DE:
    def __init__(
        self,
        problem: Problem,
        pop_size: int = 100,
        mutation: float = 0.7,
        crossover: float = 0.9,
    ) -> None:
        self.__problem: Problem = problem
        self.__pop_size: int = pop_size
        self.__mutation: float = mutation
        self.__crossover: float = crossover
        self.__population: list[Solution] = []
        self.__bests: list[Solution] = []
        self.__FEs_list: list[int] = []

    def __mutate(self) -> np.ndarray:
        new_pop = np.zeros((self.__pop_size, self.__problem.dim))
        for i in range(self.__pop_size):
            temp: list[Solution] = random.sample(self.__population, 3)
            new_pop[i] = temp[0].dec + self.__mutation * (temp[1].dec - temp[2].dec)
        return new_pop

    def __cross(self, new_pop: np.ndarray) -> None:
        for i in range(self.__pop_size):
            if random.random() > self.__crossover:
                new_pop[i] = self.__population[i].dec

    def __select(self, new_pop: np.ndarray) -> None:
        new_sol_pop = list(map(self.__problem.evaluate, new_pop))
        for i in range(self.__pop_size):
            if new_sol_pop[i].obj < self.__population[i].obj:
                self.__population[i] = new_sol_pop[i]

    def run(self, max_FEs: int = int(1e6)) -> None:
        pbar = tqdm.tqdm(total=max_FEs, unit="FEs")

        self.__bests = []
        self.__FEs_list = []
        FEs = 0
        pop_dec = self.__problem.init_population(self.__pop_size)
        self.__population: list[Solution] = list(map(self.__problem.evaluate, pop_dec))
        FEs += self.__pop_size
        self.__bests.append(min(self.__population, key=lambda x: x.obj))
        self.__FEs_list.append(FEs)

        pbar.set_description(f"Best solution: {self.__bests[-1].obj:.2e}")
        pbar.update(self.__pop_size)

        while FEs < max_FEs:
            new_pop = self.__mutate()
            self.__cross(new_pop)
            self.__select(new_pop)
            FEs += self.__pop_size
            pbar.set_description(f"Best solution: {self.__bests[-1].obj:.2e}")
            pbar.update(self.__pop_size)
            self.__bests.append(min(self.__population, key=lambda x: x.obj))
            self.__FEs_list.append(FEs)

        print()
        print(f"Best solution: {self.__bests[-1]}")

    def draw(self) -> None:
        plt.plot(self.__FEs_list, [x.obj for x in self.__bests])
        plt.xlabel("FEs")
        plt.ylabel("Objective Value")
        plt.title("Differential Evolution")
        plt.show()
