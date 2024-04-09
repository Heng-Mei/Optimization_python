"""
Date: 2024-04-08 15:17:45
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-09 17:20:36
"""

from urllib.parse import SplitResult
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

    def __mutate(self) -> list[Solution]:
        new_pop: list[Solution] = []
        for i in range(self.__pop_size):
            temp: list[Solution] = random.sample(self.__population, 3)
            new_pop.append(
                Solution(temp[0].dec + self.__mutation * (temp[1].dec - temp[2].dec))
            )
        return new_pop

    def __cross(self, new_pop: list[Solution]) -> None:
        for i in range(self.__pop_size):
            if random.random() > self.__crossover:
                new_pop[i].dec = self.__population[i].dec

    def __select(self, new_pop: list[Solution]) -> None:
        for i in range(self.__pop_size):
            if new_pop[i].obj < self.__population[i].obj:
                self.__population[i] = new_pop[i]

    def evaluate_pop(self, population: list[Solution] | None = None) -> None:
        if population is None:
            population = self.__population
        
        for solution in population:
            self.__problem.evaluate(solution)

    def run(self, max_FEs: int = int(1e6)) -> None:
        pbar = tqdm.tqdm(total=max_FEs, unit="FEs")

        self.__bests = []
        self.__FEs_list = []
        FEs = 0
        self.__population: list[Solution] = self.__problem.init_population(
            self.__pop_size
        )

        self.evaluate_pop()

        FEs += self.__pop_size
        self.__bests.append(min(self.__population, key=lambda x: x.obj))
        self.__FEs_list.append(FEs)

        pbar.set_description(f"Best solution: {self.__bests[-1].obj:.2e}")
        pbar.update(self.__pop_size)

        while FEs < max_FEs:
            new_pop = self.__mutate()
            self.__cross(new_pop)
            self.evaluate_pop(new_pop)
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
