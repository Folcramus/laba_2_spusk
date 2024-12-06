import numpy as np


def is_symmetric(A, tol=1e-10):
    """Проверка симметричности матрицы."""
    return np.allclose(A, A.T, atol=tol)


def is_positive_definite(A):
    """Проверка положительной определённости матрицы."""
    try:
        # Пытаемся выполнить разложение Холецкого.
        # Если разложение удаётся, матрица положительно определена.
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def gradient_descent(A, b, x0, tol=1e-6, max_iter=1000):
    """
    Решение СЛАУ методом наискорейшего спуска.

    Аргументы:
        A (ndarray): Матрица коэффициентов (должна быть симметричной и положительно определённой).
        b (ndarray): Вектор правой части.
        x0 (ndarray): Начальное приближение.
        tol (float): Точность (критерий завершения).
        max_iter (int): Максимальное количество итераций.

    Возвращает:
        x (ndarray): Приближённое решение.
        k (int): Количество выполненных итераций.
    """
    # Проверка условий сходимости
    if not is_symmetric(A):
        raise ValueError("Матрица A не симметрична.")
    if not is_positive_definite(A):
        raise ValueError("Матрица A не положительно определена.")

    # Инициализация переменных
    x = x0  # Текущее приближение
    r = b - A @ x  # Вектор начальной невязки (r = b - Ax)
    for k in range(max_iter):
        # Градиентный шаг
        d = r  # Направление спуска совпадает с вектором невязки

        # Вычисление оптимального шага alpha_k
        alpha = np.dot(r, r) / np.dot(d, A @ d)

        # Обновление приближения решения
        x = x + alpha * d

        # Обновление вектора невязки
        r = b - A @ x

        # Печать промежуточных результатов для отладки (можно отключить)
        print(f"Итерация {k + 1}:")
        print(f"  alpha = {alpha}")
        print(f"  x = {x}")
        print(f"  ||r|| = {np.linalg.norm(r)}")

        # Проверка критерия остановки (если невязка мала, выходим)
        if np.linalg.norm(r) < tol:
            print("Достигнута заданная точность.")
            break

    return x, k


# Пример использования
A = np.array([[1, 0.42, 0.54, 0.66],
              [0.42, 1, 0.32, 0.44],
              [0.54, 0.32, 1, 0.22],
              [0.66, 0.44, 0.22, 1]])
b = np.array([0.3, 0.5, 0.7, 0.9])
x0 = np.zeros(len(b))

try:
    x, iterations = gradient_descent(A, b, x0)
    print(f"Решение: {x}, итераций: {iterations}")
except ValueError as e:
    print(f"Ошибка: {e}")
