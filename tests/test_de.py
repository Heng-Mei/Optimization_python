"""
Date: 2024-04-08 17:36:22
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-09 10:38:31
"""
import unittest
import numpy as np
from differential_evolution import *


class TestDifferentialEvolution(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_differential_evolution(self):
        bounds = np.array(((-10, -10), (10, 10)))
        obj_fcn: Callable[[np.ndarray], float] = lambda x: x[0] ** 2 + x[1] ** 2
        problem = Problem(obj_fcn, bounds)
        de = DE(problem=problem, pop_size=100)
        de.run(50000)
        de.draw()

    def test_dimension_error(self):
        with self.assertRaises(DimensionError):
            bounds = np.array(((-10, -10), (10, 10), (1, 1)))
            obj_fcn: Callable[[np.ndarray], float] = lambda x: x[0] ** 2 + x[1] ** 2
            problem = Problem(obj_fcn, bounds)