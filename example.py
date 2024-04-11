"""
Date: 2024-04-08 15:18:15
LastEditors: Heng-Mei l888999666y@gmail.com
LastEditTime: 2024-04-11 16:13:08
"""

from typing import Callable
import numpy as np
from optimization import *

dim: int = 10

# 上下界
bounds = np.array((-10 * np.ones((1, dim)), 10 * np.ones((1, dim)))).squeeze()

# 可调用的目标函数
obj_fcn: Callable[[np.ndarray], float] = lambda x: np.matmul(x, x.T)

# 创建问题
problem = Problem(func=obj_fcn, bounds=bounds)

# 创建算法
de = DE(problem=problem, pop_size=100)

# 运行
de.solve(max_FEs=10000)

# 绘图
de.draw()

# 输出结果
de.output()
