"""
Date: 2024-04-08 15:18:15
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-08 17:36:41
"""


from typing import Callable
import numpy as np
from differential_evolution import *


def main():
    bounds = np.array(((-10, -10), (10, 10)))
    obj_fcn: Callable[[np.ndarray], float] = lambda x: x[0] ** 2 + x[1] ** 2
    problem = Problem(obj_fcn, bounds)
    de = DE(problem=problem, pop_size=100)
    de.run(50000)
    de.draw()


if __name__ == "__main__":
    main()
