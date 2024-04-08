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
