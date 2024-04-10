<!--
 * @Date: 2024-04-08 17:45:43
 * @LastEditors: Heng-Mei l888999666y@gmail.com
 * @LastEditTime: 2024-04-10 11:57:20
-->
# 基于Python3.11的优化算法实现

## 算法

- [x] [DE(Differential Evolution)，差分进化算法](optimization/algorithm/DE.py)

## 类

- [Solution](optimization/solution/Solution.py)
- [OptAlgorithm](optimization/algorithm/OptAlgorithm.py)
  > [DE(Differential Evolution)](optimization/algorithm/DE.py)
- [Problem](optimization/problem/Problem.py)

## 使用示例

```python
from typing import Callable
import numpy as np
from optimization import *

dim: int = 10

# 上下界
bounds = np.array((-10 * np.ones((1, dim)), 10 * np.ones((1, dim)))).reshape((2, dim))

# 可调用的目标函数
obj_fcn: Callable[[np.ndarray], float] = lambda x: np.matmul(x, x.T)

# 创建问题
problem = Problem(func=obj_fcn, bounds=bounds)

# 创建算法
de = DE(problem=problem, pop_size=100)

# 运行
de.run(max_FEs=10000)

# 绘图
de.draw()
```

## 迭代图

![DE迭代图](https://github.com/Heng-Mei/DE/blob/main/result/plot.png)

## 参考文献

- **DE**
  > 1. [Storn R, Price K. Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces[J]. Journal of global optimization, 1997, 11: 341-359.](https://link.springer.com/article/10.1023/a:1008202821328)
