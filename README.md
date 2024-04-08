<!--
 * @Date: 2024-04-08 17:45:43
 * @LastEditors: Heng-Mei l888999666y@gmail.com
 * @LastEditTime: 2024-04-08 17:53:55
-->
# 基于Python3.11的DE(Differential Evolution)算法实现

## 类

- Solution
- DE
- Problem

## 使用示例

```python
from typing import Callable
import numpy as np
from differential_evolution import *

# 上下界
bounds = np.array(((-10, -10), (10, 10)))

# 可调用的目标函数
obj_fcn: Callable[[np.ndarray], float] = lambda x: x[0] ** 2 + x[1] ** 2

#创建问题
problem = Problem(obj_fcn, bounds)

#创建算法
de = DE(problem=problem, pop_size=100)

#运行
de.run(50000)

#绘图
de.draw()

```
