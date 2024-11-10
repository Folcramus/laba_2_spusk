import numpy as np


def gradient_descent(A, b, x0, learning_rate, tolerance, max_iterations):
    """
  Решает систему линейных уравнений Ax = b методом наискорейшего спуска.

  Args:
    A: Матрица коэффициентов.
    b: Вектор свободных членов.
    x0: Начальное приближение решения.
    learning_rate: Шаг обучения.
    tolerance: Погрешность, при которой алгоритм останавливается.
    max_iterations: Максимальное количество итераций.

  Returns:
    Приближенное решение системы уравнений.
  """
    x = x0
    # Предварительная проверка сходимости:
    if np.linalg.cond(A) > 1e15:
        print("Матрица плохо обусловлена. Алгоритм может не сойтись.")
        return None

    for _ in range(max_iterations):
        gradient = 2 * A.T @ (A @ x - b)
        x = x - learning_rate * gradient
        if np.linalg.norm(gradient) < tolerance:
            break
    return x


# Пример использования
A = np.array([[1, 0.42, 0.54, 0.66],
              [0.42, 1, 0.32, 0.44],
              [0.54, 0.32, 1, 0.22],
              [0.66, 0.44, 0.22, 1]])
b = np.array([0.3, 0.5, 0.7, 0.9])
x0 = np.zeros(4)
learning_rate = 0.1
tolerance = 1e-6
max_iterations = 1000

x = gradient_descent(A, b, x0, learning_rate, tolerance, max_iterations)
if x is not None:
    print("Приближенное решение:", x)
else:
    print("Алгоритм не сошелся.")
