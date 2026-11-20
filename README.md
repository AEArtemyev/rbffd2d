# rbffd2d

Функции поиска соседей и вычисления весов RBF-FD с полиномиальным дополнением.
Зависит от отдельной библиотеки `rbf2d`. При расположении проектов рядом:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ../rbf2d -e .
```

Ядра находятся в `rbf2d`, функции для RBF-FD — в `rbffd2d`.

```python
import numpy as np
from rbf2d import PHS
from rbffd2d import find_neighbors, build_weights

x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
nodes = np.column_stack((x.ravel(), y.ravel()))
neighbors, weights = build_weights(nodes, PHS(3), poly_order=2)
# values — значения функции в узлах nodes
# dxx_i = values[neighbors[i]] @ weights[i][:, 3]
```

`find_neighbors(query_data, search_data=None, usage="polyOrder", ...)`
возвращает список массивов индексов с нуля. Без `search_data` поиск идёт
среди `query_data`. Режим `stencilSize` задаёт размер через `stencil_size`,
а `polyOrder` — через `poly_order` и `rho` (по умолчанию 2).
Включаются все соседи до расстояния последнего обязательного соседа плюс
`tol` (по умолчанию `1e-6`), включая саму точку.

`build_weights` возвращает соседей и список матриц весов со столбцами
`[f, dx, dy, dxx, dyy, dxy]`. В режиме `stencilSize` степень выбирается
по фактическому числу соседей, в `polyOrder` сохраняется заданная степень.
Система решается с полиномиальным дополнением. Неопределённые значения
ядра вызывают `ValueError`, вырожденная система — `numpy.linalg.LinAlgError`.

`monomial_exponents(m)` возвращает степени всех мономов суммарной степени
до `m`: сначала по суммарной степени, затем по возрастанию степени `x`.
Полиномы и функции выбора размера и степени импортируются из `rbffd2d`.

`get_poly(x, y, exps)` вычисляет мономы, а `get_poly_x`, `get_poly_y`,
`get_poly_xx`, `get_poly_yy`, `get_poly_xy` — их производные.
Строки `exps` задают степени `[a, b]` мономов `x**a * y**b`.
Последняя ось результата соответствует строкам `exps`;
остальные оси определяются broadcasting координат.

```python
from rbffd2d import get_poly, get_poly_xx

exps = [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
values = get_poly([0, 1], [0, 2], exps)  # (2, 6)
dxx = get_poly_xx(0, 0, exps)           # [0, 0, 0, 2, 0, 0]
```

`eval_monomial_deriv(exps, x0, y0, ix, iy)` вычисляет частные производные
произвольного порядка для мономов произвольной степени. Степени задаются
в `exps`, порядки производных — неотрицательными целыми `ix`, `iy`.
Последняя ось результата соответствует мономам; координаты поддерживают
broadcasting.

`get_poly*` — явные реализации для значений и производных до второго порядка.
`eval_monomial_deriv` обобщает их на произвольные порядки производных.
При соответствующих `ix`, `iy` результаты совпадают по значениям, форме
и порядку мономов. Все эти функции поддерживают мономы произвольной степени.

| Вызов | Эквивалент |
| --- | --- |
| `eval_monomial_deriv(exps, x, y, 0, 0)` | `get_poly(x, y, exps)` |
| `eval_monomial_deriv(exps, x, y, 1, 0)` | `get_poly_x(x, y, exps)` |
| `eval_monomial_deriv(exps, x, y, 0, 1)` | `get_poly_y(x, y, exps)` |
| `eval_monomial_deriv(exps, x, y, 2, 0)` | `get_poly_xx(x, y, exps)` |
| `eval_monomial_deriv(exps, x, y, 0, 2)` | `get_poly_yy(x, y, exps)` |
| `eval_monomial_deriv(exps, x, y, 1, 1)` | `get_poly_xy(x, y, exps)` |

Функция возвращает производные отдельных мономов. Для полинома нужно
умножить результат на вектор коэффициентов в том же порядке, что и `exps`:

```python
from rbffd2d import eval_monomial_deriv

# p(x, y) = 3*x**5*y**4 + 2*x**3*y**2
exps = [[5, 4], [3, 2]]
coefficients = [3, 2]
derivative = eval_monomial_deriv(exps, 2, -3, 3, 2) @ coefficients  # 77784
```

`falling_factorial(n, k)` вычисляет падающий факториал для неотрицательных
целых `n` и `k`; `n` может быть массивом. При `n < k` возвращается ноль,
при `k = 0` — единица. Обе функции импортируются из `rbffd2d`.

`get_rbf_fd_stencil_size(m, d, rho=2.5, mode="stable")` возвращает
`(n, Nm)`: число соседей и число мономов `Nm = C(m+d, d)`.
В режиме `min` число соседей равно `Nm`, в `stable` — `ceil(rho*Nm)`.

`get_poly_order(n, d, rho=2.0, mode="stable")` возвращает наибольшую степень,
для которой `C(m+d, d) <= n/rho`; режим `max` использует `n` вместо `n/rho`.
Поддерживаются размерности 2 и 3. При `n/rho < 1` в режиме `stable`
возникает `ValueError`. Название режима `stable` не гарантирует устойчивость.


Проверка:

```bash
.venv/bin/python -B -m unittest discover -s tests
```
